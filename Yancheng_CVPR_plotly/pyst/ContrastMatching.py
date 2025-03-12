from math import log10, log2
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import json
from scipy.optimize import minimize, root_scalar, minimize_scalar
from matplotlib.lines import Line2D


class ContrastConstancySpatialFreqAchTest(SyntheticTest):
    """
    The test of contrast constancy across spatial frequencies
    """

    # We are only varying the frequencies, as the ref contrasts are pre-defined!

    def __init__(self, width=256, height=256, N_freq=20, device=None):


        super().__init__(device)

        self.data_folder = 'pyst/pyst_data/contrast_constancy'

        self.width = width
        self.height = height

        self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)

        self.bkg_luminance = 10
        self.display_photometry = vvdp_display_photo_eotf(Y_peak=46.72, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([256, 256], diagonal_size_inches=5, ppd=50)

        self.ref_spatial_freq = 5
        self.ppd = self.display_geometry.get_ppd()
        self.frames = 1
        self.fps = 1

        # Get reference data
        ref_contrasts, spatial_freq, _ = self.get_gt_results()
        self.contrasts = torch.tensor(ref_contrasts, device = self.device)

        # Test data
        #self.spatial_frequencies = torch.logspace(-2, log2(25), N_freq, base=2).to(self.device)
        self.spatial_frequencies = torch.tensor(spatial_freq, device=self.device)

        # We will follow the format of previous tests
        self.N_contrast = len(self.contrasts)
        self.N_s_freq = len(self.spatial_frequencies)

        self.contrast_indices, self.s_freq_indices = torch.meshgrid(torch.arange(self.N_contrast), torch.arange(self.N_s_freq), indexing='ij')

        self.contrast_indices = self.contrast_indices.flatten()
        self.s_freq_indices = self.s_freq_indices.flatten()

        self._preview_folder = os.path.join('contrast_matching', 'spatial_frequencies')
        self._preview_as = 'image'

        self.colors = ['#C1121F', '#FB8500', '#FFB703', '#00B0A2', '#22577A', '#560BAD', '#B5179E', '#DA627D']


    def __len__(self):
        return self.N_contrast*self.N_s_freq

    def get_test_condition_parameters(self, index):

        ref_contrast = self.contrasts[self.contrast_indices[index]]
        s_freq = self.spatial_frequencies[self.s_freq_indices[index]]

        return ref_contrast, s_freq
    
    def get_test_condition(self, index):

        # Here we are simply generating the reference and test conditions to which we are matching, this is used for
        # preview, but not for the optimization! This is not similar to the other tests!

        ref_contrast, s_freq = self.get_test_condition_parameters(index)
        ref_s_freq = self.ref_spatial_freq
        Lmax = self.display_photometry.get_peak_luminance()

        XYZ_test, _ = generate_sinusoidal_grating(self.width, self.height, self.frames, self.ppd, self.fps,
                                                              ref_contrast, s_freq, self.bkg_luminance,
                                                              self.dkl_col_direction,
                                                              device=self.device)

        XYZ_reference, _ = generate_sinusoidal_grating(self.width, self.height, self.frames, self.ppd, self.fps,
                                                  ref_contrast, ref_s_freq, self.bkg_luminance,
                                                  self.dkl_col_direction,
                                                  device=self.device)

        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW",
                                               display_photometry=self.display_photometry)


    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def generate_grating(self, contrast, s_freq):
        Lmax = self.display_photometry.get_peak_luminance()
        XYZ_test, XYZ_reference = generate_sinusoidal_grating(self.width, self.height, self.frames, self.ppd, self.fps,
                                                              contrast, s_freq, self.bkg_luminance, self.dkl_col_direction,
                                                              device=self.device)
        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW",
                                               display_photometry=self.display_photometry)

    def optimization_fct(self, test_contrast, s_freq, metric_fct, ref_score_normalized, test_score_contrast_1, reverse=False, metric_max=None):
        test_contrast = 10**test_contrast
        test_video_source = self.generate_grating(torch.tensor(test_contrast, device=self.device), s_freq)
        with torch.no_grad():
            test_score, _ = metric_fct.predict_video_source(test_video_source)

        test_score_normalized = test_score/test_score_contrast_1   # here to change

        objective = ref_score_normalized - test_score_normalized

        return objective.item()

    # def optimization_fct_squared(self, test_contrast, s_freq, metric_fct, ref_score_normalized, test_score_contrast_1, reverse=False, metric_max=None):
    #     test_video_source = self.generate_grating(torch.tensor(test_contrast, device=self.device), s_freq)
    #     with torch.no_grad():
    #         test_score, _ = metric_fct.predict_video_source(test_video_source)
    #     # if reverse:
    #     #     test_score = metric_max - test_score

    #     test_score_normalized = test_score/test_score_contrast_1   # here to change

    #     objective = (ref_score_normalized - test_score_normalized)**2

    #     return objective.item()

    def run_on_test_condition(self, index, metric_fct):
        ref_contrast, s_freq = self.get_test_condition_parameters(index)
        ref_s_freq = self.ref_spatial_freq

        if s_freq.item() == ref_s_freq:
            return ref_contrast.item()

        # reverse = not metric_fct.is_lower_better()

        # Generate reference grating
        ref_video_source = self.generate_grating(ref_contrast, ref_s_freq)
        with torch.no_grad():
            ref_score, _ = metric_fct.predict_video_source(ref_video_source)   # change as well
            # if reverse:
            #     ref_score = metric_max - ref_score
        
        ref_score_normalized = ref_score

        # ref_video_source = self.generate_grating(1, ref_s_freq)
        # with torch.no_grad():
        #     ref_score_contrast_1, _ = metric_fct.predict_video_source(ref_video_source)   # change as well
        #     if reverse:
        #         ref_score_contrast_1 = metric_max - ref_score_contrast_1

        # ref_score_normalized = ref_score / ref_score_contrast_1

        # Generate test grating
        # test_video_source = self.generate_grating(1, s_freq)
        # with torch.no_grad():
        #     test_score_contrast_1, _ = metric_fct.predict_video_source(test_video_source)  # change as well!
        #     if reverse:
        #         test_score_contrast_1 = metric_max - test_score_contrast_1

        test_score_contrast_1 = 1.0

        # # Visualization
        # differences = []
        # for test_contrast in torch.logspace(-3, 0, 10):
        #     differences.append(self.optimization_fct(test_contrast.item(), s_freq, metric_fct, ref_score_normalized, test_score_contrast_1))
        #
        # print(differences)

        # Start optimization

        # res = minimize(self.optimization_fct, ref_contrast.item(), args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bounds=[(0.001, 1)])
        # res = root_scalar(self.optimization_fct, x0=ref_contrast.item(),args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bracket=[0.001, 1])
        # res = minimize_scalar(self.optimization_fct, args=(s_freq, metric_fct, ref_score, 1.0), bounds=[0.001, 1])
        # test_contrast = res.x

        try:
            res = root_scalar(self.optimization_fct, x0=ref_contrast.item(), args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bracket=[-3, 0])
            test_contrast = 10**res.root
        except:
            test_contrast = np.nan
            # try:
            #     res = minimize_scalar(self.optimization_fct_squared, args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bounds=[0.001, 1])
            #     test_contrast = res.x
            # except:
            #     test_contrast = np.nan

        return test_contrast

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as
    
    def get_ticks(self):
        #x = torch.logspace(-2, 4, 7, base=2)
        x = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        y = [0.001, 0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.spatial_frequencies[self.s_freq_indices]]

    def size(self):
        return self.N_contrast, self.N_s_freq

    def units(self):
        return ['log', 'log'], [None, 'cpd']
    
    def get_gt_results(self):
        df = json.load(open(os.path.join(self.data_folder, 'contrast_constancy_sin_5_cpd.json')))
        ref_contrasts = df['average_reference_contrast']
        spatial_frequencies = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        gt_contrasts = []
        for i in range(len(ref_contrasts)):
            gt_contrasts.append(df['ref_contrast_index_'+str(i)]['y_test_contrast_average'])

        return ref_contrasts, spatial_frequencies, gt_contrasts
    

    def plot(self, predictions, reverse_color_order=False, title=None, output_filename=None, axis=None, is_first_column=False, fontsize=12):

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        _, _, gt_contrasts = self.get_gt_results()
        gt_contrasts = np.array(gt_contrasts)

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 4))

        # Plot the results
        for i in range(len(predictions)):
            #GT
            axis.plot(x_condition[i], gt_contrasts[i], linewidth=2, linestyle='--', marker='o', color=self.colors[i])
            #Prediction
            axis.plot(x_condition[i], predictions[i], linewidth=2, marker='x', color=self.colors[i])

        axis.grid(alpha=0.4, axis='x')

        scales, units = self.units()

        if units[1]:
            axis.set_xlabel(x_condition_name+' ['+units[1]+']', fontsize=fontsize)
        else:
            axis.set_xlabel(x_condition_name, fontsize=fontsize)

        if is_first_column:
            if units[0]:
                axis.set_ylabel(y_condition_name+' ['+units[1]+']', fontsize=fontsize)
            else:
                axis.set_ylabel(y_condition_name, fontsize=fontsize)

            label = ' '.join(self.latex_name().split(' - ')[:2])
            label += '\n' + ' '.join(self.latex_name().split(' - ')[2:])

            if output_filename is None:
                axis.text(-0.28, 0.5, label, fontsize=fontsize+2, fontweight='bold', transform=axis.transAxes, ha='center', va='center', rotation=90)
        
        axis.set_xscale(scales[1])
        axis.set_yscale(scales[0])
        
        x_ticks, y_ticks = self.get_ticks()

        formatted_xticks = [f'{tick:g}' for tick in x_ticks]

        axis.set_xticks(x_ticks, formatted_xticks, fontsize=fontsize-2)

        if is_first_column:
            axis.set_yticks(y_ticks, y_ticks, fontsize=fontsize-2)
        else:
            axis.set_yticks([])

        axis.set_xlim([np.min(x_ticks), np.max(x_ticks)])
        axis.set_ylim([np.min(y_ticks), np.max(y_ticks)])

        custom_legend = [
            Line2D([0], [0], linestyle='-', marker='x', color='black', label='Metric Prediction'),
            Line2D([0], [0], linestyle='--', marker='o', color='black', label='Ground-truth')
        ]

        axis.legend(handles=custom_legend, loc=4, fontsize=fontsize-2)

        if title is not None:
            axis.set_title(title, fontsize=fontsize+2)

        if output_filename is not None:
            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        
        if axis is None:
            plt.close(fig)
    
    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""        

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        # Get Ground Truth

        y_condition, x_condition, gt_contrasts = self.get_gt_results()
        gt_contrasts = np.array(gt_contrasts)

        # Now Making the figure!

        y_condition_name = condition_names[0]
        if units_names[0] is not None:
            y_condition_name += " ["+units_names[0]+"]"

        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " ["+units_names[1]+"]"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()

        # Create the necessary variables now - Just lines basically
        traces = ''
        for i in range(len(predictions)):
            # We will start by drawing the reference 
            trace = 'var trace_ref_'+str(i)+' = { '
            x = 'x: ['
            for j, val in enumerate(x_condition):
                if j>0:
                    x += ', '
                x += str(val)
            x += ']'
            ref_contrast = gt_contrasts[i]
            y = 'y: ['
            for j, val in enumerate(ref_contrast):
                if j>0:
                    y += ', '
                y += str(val)
            y += ']'
            trace += x + ', ' + y + ", mode: 'lines+markers', marker: {color: '"+self.colors[i]+"', size:8}, line: {dash:'dot', color: '"+self.colors[i]+"', width:2},\n"
            trace += "hovertemplate: 'gt test contrast: %" + r"{y:.4f}" + "',\n"
            trace += "name: 'Contrast = "+str(np.round(y_condition[i], 4))+"', showlegend: false"
            trace += '};'

            traces += trace +'\n'

            # Now we will do that of the metrics!
            trace = 'var trace_prediction_'+str(i)+' = { '
            x = 'x: ['
            for j, val in enumerate(x_condition):
                if j>0:
                    x += ', '
                x += str(val)
            x += ']'
            prediction = predictions[i]
            y = 'y: ['
            for j, val in enumerate(prediction):
                if j>0:
                    y += ', '
                if not np.isnan(val):
                    y += str(val)
                else:
                    y+= 'null'
            y += ']'

            trace += x + ', ' + y + ", mode: 'lines+markers', marker: {symbol: 'x', color: '"+self.colors[i]+"', size:8}, line: {color: '"+self.colors[i]+"', width:2},\n"
            trace += "hovertemplate: 'metric test contrast: %" + r"{y:.4f}" + "',\n"
            trace += "name: 'Contrast = "+str(np.round(y_condition[i], 4))+"', showlegend: false"
            trace += '};'

            traces += trace +'\n'

        traces += 'var data = ['
        for i in range(len(predictions)):
            if i>0:
                traces += ', '
            traces += 'trace_ref_'+str(i)
            traces += ', trace_prediction_'+str(i)
        traces += ']\n'

        js_command = traces

        # Layout of the Figure

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = "xaxis: {type:'log', range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', range:[-3, 0], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def short_name(self):
        return 'Contrast Matching - Across Spatial Frequencies'
    
    def latex_name(self):
        return 'Contrast - Matching - Spatial - Freq.'



class ContrastConstancyColorTest(SyntheticTest):
    """
    The test of contrast constancy across cardinal color directions
    """

    # In this test the number of elements in the x-axis is defined as 3, for the 3 color directions Ach - RedGreen - Yellow Violet

    def __init__(self, width=256, height=256, N_contrasts=10, device=None):

        super().__init__(device)

        # We do not need a gt folder for this specific test!

        self.width = width
        self.height = height

        self.bkg_luminance = 21.4
        self.display_photometry = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([256, 256], diagonal_size_inches=5, ppd=60)

        self.s_freq = 1
        self.ppd = self.display_geometry.get_ppd()
        self.frames = 1
        self.fps = 1

        self.contrasts = torch.logspace(-2, log10(0.2), N_contrasts).to(self.device)
        self.dkl_col_directions = torch.tensor([[1., 0, 0],
                                                [0, 0.610649, 0],
                                                [0, 0, 4.203636]
                                                ], device=self.device)
        self.dkl_col_directions_names = np.array(['Ach', 'RG', 'YV'])

        self.N_contrasts = len(self.contrasts)
        self.N_col_directions = len(self.dkl_col_directions)

        self.contrasts_indices, self.col_direction_indices = torch.meshgrid(torch.arange(self.N_contrasts), torch.arange(self.N_col_directions), indexing='ij')

        self.contrasts_indices = self.contrasts_indices.flatten()
        self.col_direction_indices = self.col_direction_indices.flatten()

        self._preview_folder = os.path.join('contrast_matching', 'color_directions')
        self._preview_as = 'image'

        self.colors = ['#DA627D', '#B5179E', '#560BAD', '#22577A', '#00B0A2',
                      '#386641', '#FFB703', '#FB8500', '#C1121F', '#780000']

    def __len__(self):
        return self.N_contrasts * self.N_col_directions

    def get_test_condition_parameters(self, index):

        contrast = self.contrasts[self.contrasts_indices[index]]
        dkl_col_direction = self.dkl_col_directions[self.col_direction_indices[index]]

        return contrast, dkl_col_direction

    def get_test_condition(self, index):

        contrast, dkl_col_direction = self.get_test_condition_parameters(index)
        L = self.bkg_luminance
        Lmax = self.display_photometry.get_peak_luminance()
        s_freq = self.s_freq

        XYZ_test, XYZ_reference = generate_sine_wave(self.width, self.height, self.frames, self.ppd, self.fps,
                                                              contrast, s_freq, L, dkl_col_direction, device=self.device)

        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW", display_photometry=self.display_photometry)

    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def get_ticks(self):
        x = ['Ach', 'RG', 'YV']
        y = [0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Color Direction']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrasts_indices], self.dkl_col_directions_names[self.col_direction_indices.cpu().numpy()]]

    def size(self):
        return self.N_contrasts, self.N_col_directions

    def units(self):
        return ['log', 'lin'], [None, None]


    def plot(self, predictions, reverse_color_order=False, title=None, output_filename=None, axis=None, is_first_column=False, fontsize=12):


        y_condition_name = 'Metric Prediction'
        x_condition_name = 'Color Direction'

        predictions = predictions.reshape(self.size())

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 4))

        # Plot the results
        for i in range(len(predictions)):
            #Prediction
            axis.plot([1, 2, 3], predictions[i], linewidth=2, marker='o', color=self.colors[i])

        axis.grid(alpha=0.4, axis='x')

        scales, units = self.units()

        if units[1]:
            axis.set_xlabel(x_condition_name+' ['+units[1]+']', fontsize=fontsize)
        else:
            axis.set_xlabel(x_condition_name, fontsize=fontsize)

        if is_first_column:
            if units[0]:
                axis.set_ylabel(y_condition_name+' ['+units[1]+']', fontsize=fontsize)
            else:
                axis.set_ylabel(y_condition_name, fontsize=fontsize)

            label = ' '.join(self.latex_name().split(' - ')[:2])
            label += '\n' + ' '.join(self.latex_name().split(' - ')[2:])

            if output_filename is None:
                axis.text(-0.28, 0.5, label, fontsize=fontsize+2, fontweight='bold', transform=axis.transAxes, ha='center', va='center', rotation=90)
        
        #axis.set_xscale(scales[1])
        #axis.set_yscale(scales[0])
        
        x_ticks, y_ticks = self.get_ticks()

        formatted_yticks = [f'{tick:g}' for tick in y_ticks]

        axis.set_xticks([1, 2, 3], x_ticks, fontsize=fontsize-2)
        axis.tick_params(axis='y', which='major', labelsize=fontsize-2)

        #axis.set_yticks(y_ticks, formatted_yticks, fontsize=10)

        #axis.set_xlim([np.min(x_ticks), np.max(x_ticks)])
        #axis.set_ylim([np.min(y_ticks), np.max(y_ticks)])

        if not reverse_color_order:
            axis.invert_yaxis()

        if title is not None:
            axis.set_title(title, fontsize=fontsize+2)

        if output_filename is not None:
            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        
        if axis is None:
            plt.close(fig)

    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        y_condition = np.unique(conditions[0].cpu().numpy())
        #y_condition_name = condition_names[0]
        y_condition_name = 'Metric Prediction'
        if units_names[0] is not None:
            y_condition_name += " [" + units_names[0] + "]"

        x_condition = np.unique(conditions[1])
        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " [" + units_names[1] + "]"

        z_condition_name = "Metric Prediction"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()

        traces = ''
        # Here, we want to simply draw lines
        for i in range(len(predictions)):
            trace = 'var trace'+str(i)+' = { '
            x = 'x: [1, 2, 3]'
            prediction = predictions[i]
            y = 'y: ['
            for j, val in enumerate(prediction):
                if j>0:
                    y += ', '
                y += str(val)
            y += ']'

            trace += x + ', ' + y + ", mode: 'lines+markers', marker: {color: '"+self.colors[i]+"', size:8}, line: {color: '"+self.colors[i]+"', width:2},\n"
            trace += "hovertemplate: 'Prediction: %" + r"{y:.2f}" + "',\n"
            trace += "name: 'Contrast = "+str(np.round(y_condition[i], 2))+"', showlegend: false"
            trace += '};'

            traces += trace +'\n'

        traces += 'var data = ['
        for i in range(len(predictions)):
            if i>0:
                traces += ', '
            traces += 'trace'+str(i)
        traces += ']\n'

        # If reverse, we want to reverse the whole y axis!

        # We also want to find a way to include ground truth as well!

        # Layout of the Figure

        js_command = traces

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i > 0:
                x += ", "
            x += "'"+el+"'"
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i > 0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = ("xaxis: {tickvals: [1, 2, 3], ticktext: " + x + ", title:'" + x_condition_name +
                     "'}, yaxis: {" )
        if not reverse_color_order:
            axes_info += "autorange: 'reversed', "
        axes_info += "title:'" + y_condition_name + "'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '" + title + "', y: 0.95}, width:450, height:400, " + axes_info + ",\n" + edge_info + ",\n" + margin_info + "\n};"

        if fig_id:
            js_command += "\nPlotly.newPlot('" + fig_id + "', data, layout);\n\n"

        return js_command

    def short_name(self):
        return 'Contrast Matching - Across Cardinal Color Directions'
    
    def latex_name(self):
        return 'Contrast - Matching - Color - Direction'

class ContrastConstancySpatialFreqAchNoiseTest(SyntheticTest):
    """
    The test of contrast constancy across spatial frequencies - using band limited noise instead!
    """

    # We are only varying the frequencies, as the ref contrasts are pre-defined!

    def __init__(self, width=256, height=256, N_freq=20, device=None):


        super().__init__(device)

        self.data_folder = 'pyst/pyst_data/contrast_constancy'

        self.width = width
        self.height = height

        self.dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)

        self.bkg_luminance = 10
        self.display_photometry = vvdp_display_photo_eotf(Y_peak=46.72, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([256, 256], diagonal_size_inches=5, ppd=50)

        self.ref_spatial_freq = torch.tensor(5., device=self.device)
        self.ppd = self.display_geometry.get_ppd()
        self.frames = 1
        self.fps = 1

        # Get reference data
        ref_contrasts, spatial_freq, _ = self.get_gt_results()
        self.contrasts = torch.tensor(ref_contrasts, device = self.device)

        # Test data
        #self.spatial_frequencies = torch.logspace(-2, log2(25), N_freq, base=2).to(self.device)
        self.spatial_frequencies = torch.tensor(spatial_freq, device=self.device)

        # We will follow the format of previous tests
        self.N_contrast = len(self.contrasts)
        self.N_s_freq = len(self.spatial_frequencies)

        self.contrast_indices, self.s_freq_indices = torch.meshgrid(torch.arange(self.N_contrast), torch.arange(self.N_s_freq), indexing='ij')

        self.contrast_indices = self.contrast_indices.flatten()
        self.s_freq_indices = self.s_freq_indices.flatten()

        self._preview_folder = os.path.join('contrast_matching', 'spatial_frequencies_noise')
        self._preview_as = 'image'

        self.colors = ['#C1121F', '#FB8500', '#FFB703', '#00B0A2', '#22577A', '#560BAD', '#B5179E', '#DA627D']


    def __len__(self):
        return self.N_contrast*self.N_s_freq

    def get_test_condition_parameters(self, index):

        ref_contrast = self.contrasts[self.contrast_indices[index]]
        s_freq = self.spatial_frequencies[self.s_freq_indices[index]]

        return ref_contrast, s_freq
    
    def get_test_condition(self, index):

        # Here we are simply generating the reference and test conditions to which we are matching, this is used for
        # preview, but not for the optimization! This is not similar to the other tests!

        ref_contrast, s_freq = self.get_test_condition_parameters(index)
        ref_s_freq = self.ref_spatial_freq
        Lmax = self.display_photometry.get_peak_luminance()

        XYZ_test, _ = generate_band_limited_noise(self.width, self.height, self.ppd,
                                                              ref_contrast, s_freq, 1.0 , self.bkg_luminance,
                                                              self.dkl_col_direction, self.frames,
                                                              device=self.device)

        XYZ_reference, _ = generate_band_limited_noise(self.width, self.height, self.ppd,
                                                  ref_contrast, ref_s_freq, 1.0, self.bkg_luminance,
                                                  self.dkl_col_direction, self.frames,
                                                  device=self.device)

        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW",
                                               display_photometry=self.display_photometry)


    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def generate_grating(self, contrast, s_freq):
        Lmax = self.display_photometry.get_peak_luminance()
        XYZ_test, XYZ_reference = generate_band_limited_noise(self.width, self.height, self.ppd,
                                                              contrast, s_freq, 1.0, self.bkg_luminance, self.dkl_col_direction, self.frames,
                                                              device=self.device)
        test_condition = lin2srgb(XYZ2RGB709_nd(XYZ_test / Lmax).unsqueeze(0))
        reference_condition = lin2srgb(XYZ2RGB709_nd(XYZ_reference / Lmax).unsqueeze(0))

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW",
                                               display_photometry=self.display_photometry)

    def optimization_fct(self, test_contrast, s_freq, metric_fct, ref_score_normalized, test_score_contrast_1, reverse=False, metric_max=None):
        test_contrast = 10**test_contrast
        test_video_source = self.generate_grating(torch.tensor(test_contrast, device=self.device), s_freq)
        with torch.no_grad():
            test_score, _ = metric_fct.predict_video_source(test_video_source)

        test_score_normalized = test_score/test_score_contrast_1   # here to change

        objective = ref_score_normalized - test_score_normalized

        return objective.item()

    # def optimization_fct_squared(self, test_contrast, s_freq, metric_fct, ref_score_normalized, test_score_contrast_1, reverse=False, metric_max=None):
    #     test_video_source = self.generate_grating(torch.tensor(test_contrast, device=self.device), s_freq)
    #     with torch.no_grad():
    #         test_score, _ = metric_fct.predict_video_source(test_video_source)
    #     # if reverse:
    #     #     test_score = metric_max - test_score

    #     test_score_normalized = test_score/test_score_contrast_1   # here to change

    #     objective = (ref_score_normalized - test_score_normalized)**2

    #     return objective.item()

    def run_on_test_condition(self, index, metric_fct):
        ref_contrast, s_freq = self.get_test_condition_parameters(index)
        ref_s_freq = self.ref_spatial_freq

        if s_freq.item() == ref_s_freq:
            return ref_contrast.item()


        # Generate reference grating
        ref_video_source = self.generate_grating(ref_contrast, ref_s_freq)
        with torch.no_grad():
            ref_score, _ = metric_fct.predict_video_source(ref_video_source)   # change as well
        
        ref_score_normalized = ref_score

        test_score_contrast_1 = 1.0

        # # Visualization
        # differences = []
        # for test_contrast in torch.logspace(-3, 0, 10):
        #     differences.append(self.optimization_fct(test_contrast.item(), s_freq, metric_fct, ref_score_normalized, test_score_contrast_1))
        
        # print(differences)

        # Start optimization

        try:
            res = root_scalar(self.optimization_fct, x0=ref_contrast.item(), args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bracket=[-3, 0])
            test_contrast = 10**res.root
        except:
            test_contrast = np.nan
            # try:
            #     res = minimize_scalar(self.optimization_fct_squared, args=(s_freq, metric_fct, ref_score_normalized, test_score_contrast_1), bounds=[0.001, 1])
            #     test_contrast = res.x
            # except:
            #     test_contrast = np.nan

        return test_contrast

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as
    
    def get_ticks(self):
        #x = torch.logspace(-2, 4, 7, base=2)
        x = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        y = [0.001, 0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Contrast', 'Spatial Frequency']

    def get_rows_conditions(self):
        return [self.contrasts[self.contrast_indices], self.spatial_frequencies[self.s_freq_indices]]

    def size(self):
        return self.N_contrast, self.N_s_freq

    def units(self):
        return ['log', 'log'], [None, 'cpd']
    
    def get_gt_results(self):
        df = json.load(open(os.path.join(self.data_folder, 'contrast_constancy_sin_5_cpd.json')))
        ref_contrasts = df['average_reference_contrast']
        spatial_frequencies = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
        gt_contrasts = []
        for i in range(len(ref_contrasts)):
            gt_contrasts.append(df['ref_contrast_index_'+str(i)]['y_test_contrast_average'])

        return ref_contrasts, spatial_frequencies, gt_contrasts
    

    def plot(self, predictions, reverse_color_order=False, title=None, output_filename=None, axis=None, is_first_column=False, fontsize=12):

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        _, _, gt_contrasts = self.get_gt_results()
        gt_contrasts = np.array(gt_contrasts)

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 4))

        # Plot the results
        for i in range(len(predictions)):
            #GT
            axis.plot(x_condition[i], gt_contrasts[i], linewidth=2, linestyle='--', marker='o', color=self.colors[i])
            #Prediction
            axis.plot(x_condition[i], predictions[i], linewidth=2, marker='x', color=self.colors[i])

        axis.grid(alpha=0.4, axis='x')

        scales, units = self.units()

        if units[1]:
            axis.set_xlabel(x_condition_name+' ['+units[1]+']', fontsize=fontsize)
        else:
            axis.set_xlabel(x_condition_name, fontsize=fontsize)

        if is_first_column:
            if units[0]:
                axis.set_ylabel(y_condition_name+' ['+units[1]+']', fontsize=fontsize)
            else:
                axis.set_ylabel(y_condition_name, fontsize=fontsize)

            label = ' '.join(self.latex_name().split(' - ')[:2])
            label += '\n' + ' '.join(self.latex_name().split(' - ')[2:])

            if output_filename is None:
                axis.text(-0.28, 0.5, label, fontsize=fontsize+2, fontweight='bold', transform=axis.transAxes, ha='center', va='center', rotation=90)
        
        axis.set_xscale(scales[1])
        axis.set_yscale(scales[0])
        
        x_ticks, y_ticks = self.get_ticks()

        formatted_xticks = [f'{tick:g}' for tick in x_ticks]

        axis.set_xticks(x_ticks, formatted_xticks, fontsize=fontsize-2)

        if is_first_column:
            axis.set_yticks(y_ticks, y_ticks, fontsize=fontsize-2)
        else:
            axis.set_yticks([])

        axis.set_xlim([np.min(x_ticks), np.max(x_ticks)])
        axis.set_ylim([np.min(y_ticks), np.max(y_ticks)])

        custom_legend = [
            Line2D([0], [0], linestyle='-', marker='x', color='black', label='Metric Prediction'),
            Line2D([0], [0], linestyle='--', marker='o', color='black', label='Ground-truth')
        ]

        axis.legend(handles=custom_legend, loc=4, fontsize=fontsize-2)

        if title is not None:
            axis.set_title(title, fontsize=fontsize+2)

        if output_filename is not None:
            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        
        if axis is None:
            plt.close(fig)
    
    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""        

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()

        # Get Ground Truth

        y_condition, x_condition, gt_contrasts = self.get_gt_results()
        gt_contrasts = np.array(gt_contrasts)

        # Now Making the figure!

        y_condition_name = condition_names[0]
        if units_names[0] is not None:
            y_condition_name += " ["+units_names[0]+"]"

        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " ["+units_names[1]+"]"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()

        # Create the necessary variables now - Just lines basically
        traces = ''
        for i in range(len(predictions)):
            # We will start by drawing the reference 
            trace = 'var trace_ref_'+str(i)+' = { '
            x = 'x: ['
            for j, val in enumerate(x_condition):
                if j>0:
                    x += ', '
                x += str(val)
            x += ']'
            ref_contrast = gt_contrasts[i]
            y = 'y: ['
            for j, val in enumerate(ref_contrast):
                if j>0:
                    y += ', '
                y += str(val)
            y += ']'
            trace += x + ', ' + y + ", mode: 'lines+markers', marker: {color: '"+self.colors[i]+"', size:8}, line: {dash:'dot', color: '"+self.colors[i]+"', width:2},\n"
            trace += "hovertemplate: 'gt test contrast: %" + r"{y:.4f}" + "',\n"
            trace += "name: 'Contrast = "+str(np.round(y_condition[i], 4))+"', showlegend: false"
            trace += '};'

            traces += trace +'\n'

            # Now we will do that of the metrics!
            trace = 'var trace_prediction_'+str(i)+' = { '
            x = 'x: ['
            for j, val in enumerate(x_condition):
                if j>0:
                    x += ', '
                x += str(val)
            x += ']'
            prediction = predictions[i]
            y = 'y: ['
            for j, val in enumerate(prediction):
                if j>0:
                    y += ', '
                if not np.isnan(val):
                    y += str(val)
                else:
                    y+= 'null'
            y += ']'

            trace += x + ', ' + y + ", mode: 'lines+markers', marker: {symbol: 'x', color: '"+self.colors[i]+"', size:8}, line: {color: '"+self.colors[i]+"', width:2},\n"
            trace += "hovertemplate: 'metric test contrast: %" + r"{y:.4f}" + "',\n"
            trace += "name: 'Contrast = "+str(np.round(y_condition[i], 4))+"', showlegend: false"
            trace += '};'

            traces += trace +'\n'

        traces += 'var data = ['
        for i in range(len(predictions)):
            if i>0:
                traces += ', '
            traces += 'trace_ref_'+str(i)
            traces += ', trace_prediction_'+str(i)
        traces += ']\n'

        js_command = traces

        # Layout of the Figure

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_tick_vals):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        axes_info = "xaxis: {type:'log', range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', range:[-3, 0], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def short_name(self):
        return 'Contrast Matching - Across Spatial Frequencies - Band Limited Noise'
    
    def latex_name(self):
        return 'Contrast - Matching - Spatial - Freq. Noise'