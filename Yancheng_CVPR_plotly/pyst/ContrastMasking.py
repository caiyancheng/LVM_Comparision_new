from math import log10
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import pandas as pd
import json


class ContrastMakingTest(SyntheticTest):
    """
    Abstract class for all synthetic tests relying on a gabor masker!
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, width, height, test_channel, masker_channel, display_photometry, display_geometry, frames, fps, test_contrasts, masker_contrasts, spatial_frequencies=2, temporal_frequencies=0,
                 bkg_luminances=21.4, masker_orientations=0, test_orientations=0, test_sizes=2, phase_masking='coherent', N_multiplier=10, is_alignmenet_score=False, device=None):


        super().__init__(device)

        self.width = width
        self.height = height

        self.test_channel = test_channel
        self.masker_channel = masker_channel

        self.phase_masking = phase_masking

        if test_channel == 'Ach':
            self.test_dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif test_channel == 'RG':
            self.test_dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif test_channel == 'YV':
            self.test_dkl_col_direction = torch.tensor([0, 0, 4.203636], device=self.device)
        
        if masker_channel == 'Ach':
            self.masker_dkl_col_direction = torch.tensor([1., 0, 0], device=self.device)
        elif masker_channel == 'RG':
            self.masker_dkl_col_direction = torch.tensor([0, 0.610649, 0], device=self.device)
        elif masker_channel == 'YV':
            self.masker_dkl_col_direction = torch.tensor([0, 0, 4.203636], device=self.device)
        
        self.display_photometry = display_photometry
        self.display_geometry = display_geometry

        self.ppd = display_geometry.get_ppd()
        self.frames = frames
        self.fps = fps

        self.is_alignmenet_score = is_alignmenet_score
        if self.is_alignmenet_score:
            self.test_contrasts = torch.logspace(log10(0.5), log10(2), N_multiplier).to(self.device)
        else:
            self.test_contrasts = test_contrasts

        self.masker_contrasts = masker_contrasts

        self.spatial_frequencies =  torch.tensor([spatial_frequencies], device=self.device) if isinstance(spatial_frequencies, (int, float)) else spatial_frequencies.to(self.device)
        self.temporal_frequencies = torch.tensor([temporal_frequencies], device=self.device) if isinstance(temporal_frequencies, (int, float)) else temporal_frequencies.to(self.device)
        self.bkg_luminances = torch.tensor([bkg_luminances], device=self.device) if isinstance(bkg_luminances, (int, float)) else bkg_luminances.to(self.device)

        self.masker_orientations = torch.tensor([masker_orientations], device=self.device) if isinstance(masker_orientations, (int, float)) else masker_orientations.to(self.device)
        self.test_orientations = torch.tensor([test_orientations], device=self.device) if isinstance(test_orientations, (int, float)) else test_orientations.to(self.device)

        self.test_sizes = torch.tensor([test_sizes], device=self.device) if isinstance(test_sizes, (int, float)) else test_sizes.to(self.device)

        self.N_test_contrasts = len(self.test_contrasts)
        self.N_masker_contrasts = len(self.masker_contrasts)
        self.N_s_freq = len(self.spatial_frequencies)
        self.N_t_freq = len(self.temporal_frequencies)
        self.N_L = len(self.bkg_luminances)
        self.N_test_orien = len(self.test_orientations)
        self.N_masker_orien = len(self.masker_orientations)
        self.N_size = len(self.test_sizes)

        self.test_contrast_indices, self.masker_contrast_indices, self.s_freq_indices, self.t_freq_indices, self.L_indices, self.test_orien_indices, self.masker_orien_indices, self.size_indices = torch.meshgrid(torch.arange(self.N_test_contrasts), torch.arange(self.N_masker_contrasts), torch.arange(self.N_s_freq), torch.arange(self.N_t_freq), torch.arange(self.N_L), torch.arange(self.N_test_orien), torch.arange(self.N_masker_orien), torch.arange(self.N_size), indexing='ij')

        self.test_contrast_indices = self.test_contrast_indices.flatten()
        self.masker_contrast_indices = self.masker_contrast_indices.flatten()
        self.s_freq_indices = self.s_freq_indices.flatten()
        self.t_freq_indices = self.t_freq_indices.flatten()
        self.L_indices = self.L_indices.flatten()
        self.test_orien_indices = self.test_orien_indices.flatten()
        self.masker_orien_indices = self.masker_orien_indices.flatten()
        self.size_indices = self.size_indices.flatten()

        self._preview_folder = 'contrast_masking'
        self._preview_as = None

        self.data_folder = 'pyst/pyst_data/contrast_masking'

        if self.is_alignmenet_score:
            self.x_variable = self.masker_contrasts
            self.x_variable_indices = self.masker_contrast_indices

            self.x_gt, self.y_gt = self.get_contrast_masking_results()


    def __len__(self):
        return self.N_test_contrasts*self.N_masker_contrasts*self.N_s_freq*self.N_t_freq*self.N_L*self.N_test_orien*self.N_masker_orien*self.N_size

    @abc.abstractmethod
    def get_contrast_masking_results(self):
        pass

    def get_test_condition_parameters(self, index):

        if self.is_alignmenet_score:
            multiplier = self.test_contrasts[self.test_contrast_indices[index]]
            x_var = self.x_variable[self.x_variable_indices[index]]
            """ Using both the multiplier and the x_var_index, we can identify the contrast to be used!"""
            sensitivity_gt = 1 / np.interp(x_var.cpu(), self.x_gt, self.y_gt)
            test_contrast = 1 / (sensitivity_gt * multiplier)
        else:
            test_contrast = self.test_contrasts[self.test_contrast_indices[index]]

        masker_contrast = self.masker_contrasts[self.masker_contrast_indices[index]]

        s_freq = self.spatial_frequencies[self.s_freq_indices[index]]
        t_freq = self.temporal_frequencies[self.t_freq_indices[index]]
        L = self.bkg_luminances[self.L_indices[index]]

        test_orien = self.test_orientations[self.test_orien_indices[index]]
        masker_orien = self.masker_orientations[self.masker_orien_indices[index]]

        size = self.test_sizes[self.size_indices[index]]

        return test_contrast, masker_contrast, s_freq, t_freq, L, test_orien, masker_orien, size
    
    def get_test_condition(self, index):

        test_contrast, masker_contrast, s_freq, t_freq, L, test_orien, masker_orien, size = self.get_test_condition_parameters(index)
        Lmax = self.display_photometry.get_peak_luminance()

        if self.phase_masking == 'coherent':
            XYZ_target, XYZ_masker = generate_gabor_mask(self.width, self.height, self.ppd, test_contrast, masker_contrast, s_freq, test_orien, masker_orien, L, self.test_dkl_col_direction, self.masker_dkl_col_direction, size, device=self.device)
        else: 
            XYZ_target, XYZ_masker = generate_flat_noise_mask(self.width, self.height, self.ppd, test_contrast, masker_contrast, s_freq, test_orien, L, self.test_dkl_col_direction, self.masker_dkl_col_direction, size, device=self.device)

        test_patch = XYZ2RGB709_nd(XYZ_target / Lmax).unsqueeze(0)
        masker = XYZ2RGB709_nd(XYZ_masker / Lmax).unsqueeze(0)

        test_condition = lin2srgb(test_patch + masker)
        reference_condition = lin2srgb(masker)

        return video_source.video_source_array(test_condition, reference_condition, self.fps, dim_order="BCFHW", display_photometry=self.display_photometry)

    def plot(self, predictions, reverse_color_order=False, output_filename=None, title=None, axis=None, is_first_column=False, fontsize=12):


        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        cmap = 'viridis'

        if reverse_color_order:
            cmap = 'viridis_r'

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 4))

        contour = axis.contour(x_condition, y_condition, predictions, 30, cmap=cmap)

        # Plot the csf results

        x_gt, y_gt = self.get_contrast_masking_results()

        axis.plot(x_gt, y_gt, color='red', linewidth=2, marker='o', mfc='none', label='Ground-truth')

        #Figure parameters

        colorbar = plt.colorbar(contour, ax=axis)
        if reverse_color_order:
            colorbar.ax.invert_yaxis()

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
                axis.text(-0.35, 0.5, label, fontsize=fontsize+2, fontweight='bold', transform=axis.transAxes, ha='center', va='center', rotation=90)
        
        axis.set_xscale(scales[1])
        axis.set_yscale(scales[0])
        
        x_ticks, y_ticks = self.get_ticks()

        formatted_xticks = [f'{tick:g}' for tick in x_ticks]

        axis.set_xticks(x_ticks, formatted_xticks, fontsize=fontsize-2)
        if is_first_column:
            axis.set_yticks(y_ticks, y_ticks, fontsize=fontsize-2)
        else:
            axis.set_yticks([])

        axis.set_xlim([np.min(x_condition), np.max(x_condition)])
        axis.set_ylim([np.min(y_condition), np.max(y_condition)])

        axis.legend(loc=4, fontsize=fontsize-2)

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
        y_condition_name = condition_names[0]
        if units_names[0] is not None:
            y_condition_name += " ["+units_names[0]+"]"

        x_condition = np.unique(conditions[1].cpu().numpy())
        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " ["+units_names[1]+"]"

        z_condition_name = "Metric Prediction"

        predictions = predictions.reshape(self.size())

        x_tick_vals, y_tick_vals = self.get_ticks()

        z = "["

        for i, row in enumerate(predictions):
            if i>0:
                z += ", "
            z += "["
            for j, el in enumerate(row):
                if j > 0:
                    z += ", "
                if not np.isnan(el):
                    z += str(el)
                else:
                    z += 'null'
            z += "]"

        z += "]"

        x = "["
        for i, el in enumerate(x_condition):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_condition):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        # Contour plot data

        js_command += "var Contourdata = {\n"
        js_command += "z: " + z + ",\nx: " + x + ",\ny: " + y

        extra_params = "type: 'contour',\ncolorscale: 'Viridis',\nncontours: 30,\nline:{width:2},\n"
        if reverse_color_order:
            extra_params+= "reversescale: true,\n"

        extra_params += "hovertemplate: '"+x_condition_name+": %"+r"{x:.4f}"+"<br>"+y_condition_name+": %"+r"{y:.4f}"+"<br>"+z_condition_name+": %"+r"{z:.4f}"+"<extra></extra>',\n"
        
        extra_params += r"contours:{coloring:'lines'}"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        # Ground Truth Data 

        x_gt, y_gt = self.get_contrast_masking_results()

        x = "["
        for i, el in enumerate(x_gt):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        y = "["
        for i, el in enumerate(y_gt):
            if i>0:
                y += ", "
            y += str(el)
        y += "]"

        js_command += "\nvar Linedata = {\n"
        js_command += "x: " + x + ",\ny: " + y

        extra_params = "mode: 'lines+markers',\ntype: 'scatter',\nline:{color: 'red', width:2},marker:{symbol:'circle-open', size:8},\nname: 'Human Results'"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        js_command += "\nvar data = [Contourdata, Linedata]"

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

        axes_info = "xaxis: {type:'log', range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', range:["+str(log10(np.min(y_condition)))+","+str(log10(np.max(y_condition)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command



    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def short_name(self):
        return 'Contrast Masking Test'


class ContrastMakingCoherentAchTest(ContrastMakingTest):

    def __init__(self, width=420, height=300, N_test_contrasts=20, N_masker_contrasts=20, is_alignmenet_score=False, device=None):
        
        test_channel = 'Ach'
        masker_channel = 'Ach'

        phase_masking = 'coherent'

        frames = 1
        fps = 1

        test_contrasts = torch.logspace(np.log10(0.005), np.log10(0.5), N_test_contrasts)
        masker_contrasts = torch.logspace(np.log10(0.005), np.log10(0.5), N_masker_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=150, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([420, 300], diagonal_size_inches=9.6, ppd=60)

        super().__init__(width, height, test_channel, masker_channel, dm, gm, frames, fps, test_contrasts, masker_contrasts, 
                         spatial_frequencies=2, bkg_luminances=32, test_sizes=0.5, phase_masking=phase_masking, is_alignmenet_score=is_alignmenet_score, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'coherent_ach')
        self._preview_as = 'image'

    def get_contrast_masking_results(self):

        df = pd.read_csv(os.path.join(self.data_folder, 'foley_data.csv'))
        df = df[(df['orientation'] == 0) & (df['c_mask'] != 0)]
        x = 10 ** (df['c_mask'].to_numpy()/20)
        y = 10 ** (df['c_target'].to_numpy()/20)
        
        return x, y

    def get_row_header(self):
        return ['Test Contrast', 'Masker Contrast']

    def get_rows_conditions(self):
        return [self.test_contrasts[self.test_contrast_indices], self.masker_contrasts[self.masker_contrast_indices]]

    def size(self):
        return self.N_test_contrasts, self.N_masker_contrasts

    def get_ticks(self):
        x = [0.01, 0.1]
        y = [0.01, 0.1]

        return x, y

    def units(self):
        return ['log', 'log'], [None, None]

    def short_name(self):
        return 'Contrast Masking - Phase Coherent Masking'
    
    def latex_name(self):
        return 'Contrast - Masking - Phase - Coherent'


class ContrastMakingIncoherentAchTest(ContrastMakingTest):

    def __init__(self, width=300, height=300, N_test_contrasts=20, N_masker_contrasts=20, is_alignmenet_score=False, device=None):
        
        test_channel = 'Ach'
        masker_channel = 'Ach'

        phase_masking = 'incoherent'

        frames = 1
        fps = 1

        test_contrasts = torch.logspace(np.log10(0.005), np.log10(0.5), N_test_contrasts)
        masker_contrasts = torch.logspace(np.log10(0.005), np.log10(0.5), N_masker_contrasts)

        dm = vvdp_display_photo_eotf(Y_peak=173, contrast=1000, source_colorspace='sRGB')
        gm = vvdp_display_geometry([300, 300], diagonal_size_inches=7.8, ppd=60)

        super().__init__(width, height, test_channel, masker_channel, dm, gm, frames, fps, test_contrasts, masker_contrasts, 
                         spatial_frequencies=1.2, bkg_luminances=37, test_sizes=0.8, phase_masking=phase_masking, is_alignmenet_score=is_alignmenet_score, device=device)

        self._preview_folder = os.path.join(self._preview_folder, 'incoherent_ach')
        self._preview_as = 'image'

    def get_contrast_masking_results(self):

        df = json.load(open(os.path.join(self.data_folder, 'contrast_masking_data_gabor_on_noise.json')))
        x = df['mask_contrast_list']
        y = df['test_contrast_list']
        
        return x, y

    def get_row_header(self):
        return ['Test Contrast', 'Masker Contrast']

    def get_rows_conditions(self):
        return [self.test_contrasts[self.test_contrast_indices], self.masker_contrasts[self.masker_contrast_indices]]

    def size(self):
        return self.N_test_contrasts, self.N_masker_contrasts

    def get_ticks(self):
        x = [0.01, 0.1]
        y = [0.01, 0.1]

        return x, y

    def units(self):
        return ['log', 'log'], [None, None]

    def short_name(self):
        return 'Contrast Masking - Phase Incoherent Masking'
    
    def latex_name(self):
        return 'Contrast - Masking - Phase - Incoherent'