from math import log10
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
from pyst.utils import *
import json


class ContrastDetectionTest(SyntheticTest):
    """
    Abstract class for all synthetic tests relying on gabor patches
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, device=None):


        super().__init__(device)

        self.csf_data_folder = 'CVPR_ploting/gt_json'
        self_preview_folder = None


    def __len__(self):
        return 25

    @abc.abstractmethod
    def get_csf_results(self):
        pass

    @abc.abstractmethod
    def get_ticks(self):
        pass

    @abc.abstractmethod
    def preview_filepath(self):
        pass

    
    def plotly(self, x_condition, y_condition, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""

        condition_names = self.get_row_header()

        y_condition = 1 / np.unique(y_condition)
        y_condition_name = condition_names[0]

        x_condition = np.unique(x_condition)
        x_condition_name = condition_names[1]

        z_condition_name = "Latent code angular distance"

        predictions = np.transpose(np.array(predictions))

        x_tick_vals, y_tick_vals = self.get_ticks()

        z = "["

        for i, row in enumerate(predictions):
            if i>0:
                z += ", "
            z += "["
            for j, el in enumerate(row):
                if j > 0:
                    z += ", "
                z += str(el)
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

        extra_params = "zmin: 0, zmax: 1, type: 'contour',\ncolorscale: 'matter',\nncontours: 100,\nline:{width:2},\n"
        #if reverse_color_order:
        #extra_params+= "reversescale: true,\n"

        extra_params += "hovertemplate: '"+x_condition_name+": %"+r"{x:.2f}"+"<br>"+y_condition_name+": %"+r"{y:.2f}"+"<br>"+z_condition_name+": %"+r"{z:.2f}"+"<extra></extra>',\n"
        
        extra_params += r"contours:{coloring:'heatmap'}"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        # Ground Truth Data 

        x_gt, y_gt = self.get_csf_results()

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

        extra_params = "mode: 'lines',\ntype: 'scatter',\nline:{color: 'red', width:2},marker:{symbol:'circle-open', size:8},\nname: 'CastleCSF'"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        js_command += "\nvar data = [Contourdata, Linedata]"

        # Layout of the Figure

        # Get ticks

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

        axes_info = "xaxis: {type:'log', showgrid: false, range: ["+str(log10(np.min(x_condition)))+","+str(log10(np.max(x_condition)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', showgrid: false, range:["+str(log10(np.min(y_condition)))+","+str(log10(np.max(y_condition)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def get_preview_folder(self):
        return self._preview_folder

    def short_name(self):
        return 'Contrast Detection'


class ContrastDetectionSpatialFreqAchGaborTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_SpF_Gabor_ach')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_rho_sensitivity_data.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['rho_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.5, 1, 2, 4, 8, 16, 32]
        y = [1, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Spatial Frequency (cpd)']

    def get_rows_conditions(self):
        return ['rho_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Contrast Detection - Spatial Frequency - Achromatic Gabor'
    
    def preview_filepath(self):
        return "Gabor_SpF_Contrast_supplementary.png"

class ContrastDetectionSpatialFreqAchNoiseTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_SpF_Noise_ach')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_rho_sensitivity_data_band_lim_noise.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['rho_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.5, 1, 2, 4, 8, 16, 32]
        y = [1, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Spatial Frequency (cpd)']

    def get_rows_conditions(self):
        return ['rho_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Contrast Detection - Spatial Frequency - Achromatic Noise'
    
    def preview_filepath(self):
        return "Noise_SpF_Contrast_supplementary.png"

class ContrastDetectionSpatialFreqRGGaborTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_SpF_Gabor_RG')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_rho_sensitivity_data_RG.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['rho_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.5, 1, 2, 4, 8, 16, 32]
        y = [5, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Spatial Frequency (cpd)']
    
    def get_rows_conditions(self):
        return ['rho_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Contrast Detection - Spatial Frequency - Red-Green Gabor'

    def preview_filepath(self):
        return "Gabor_SpF_Contrast_RG_supplementary.png"

class ContrastDetectionSpatialFreqYVGaborTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_SpF_Gabor_YV')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_rho_sensitivity_data_YV.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['rho_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.5, 1, 2, 4, 8, 16, 32]
        y = [5, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Spatial Frequency (cpd)']

    def get_rows_conditions(self):
        return ['rho_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def short_name(self):
        return 'Contrast Detection - Spatial Frequency - Yellow-Violet Gabor'

    def preview_filepath(self):
        return "Gabor_SpF_Contrast_YV_supplementary.png"

class ContrastDetectionLuminanceTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_luminance')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_luminance_sensitivity_data.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['luminance_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.1, 1, 10, 100]
        y = [1, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Luminance (cd/m²)']

    def get_rows_conditions(self):
        return ['L_b_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cd/m$^2$']

    def short_name(self):
        return 'Contrast Detection - Luminance'
    
    def preview_filepath(self):
        return "Gabor_Luminance_Contrast_supplementary.png"

class ContrastDetectionAreaTest(ContrastDetectionTest):

    def __init__(self, device=None):

        super().__init__(device=device)

        self._preview_folder = os.path.join('test_contrast_detection_area')


    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'castleCSF_area_sensitivity_data.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        x = csf_results['area_list']
        y = csf_results['sensitivity_list']

        return x, y

    def get_ticks(self):
        x = [0.1, 1]
        y = [1, 10, 100, 1000]

        return x, y

    def get_row_header(self):
        return ['Sensitivity', 'Area (degree²)']
    
    def get_rows_conditions(self):
        return ['area_matrix', 'contrast_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'degree$^2$']

    def short_name(self):
        return 'Contrast Detection - Area'
    
    def preview_filepath(self):
        return "Gabor_Area_Contrast_supplementary.png"