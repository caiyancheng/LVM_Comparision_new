from math import log10
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
from pyst.utils import *
import json


class ContrastMatchingTest(SyntheticTest):
    """
    Abstract class for all synthetic tests relying on gabor patches
    """

    # For now we may not need these informations! It is not as important

    def __init__(self, device=None):


        super().__init__(device)

        self.csf_data_folder = 'CVPR_ploting/gt_json'
        self._preview_folder = os.path.join('test_contrast_matching_cos_scale_solve')
        self.colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'indigo', 'violet']


    def __len__(self):
        return 25

    
    def plotly(self, conditions, reference_contrast_list, rho_test_list, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""

        condition_names = self.get_row_header()
        y_condition_name = condition_names[0]
        x_condition_name = condition_names[1]
        z_condition_name = "Latent code dissimilarity"

        x_tick_vals, y_tick_vals = self.get_ticks()

        human_result_data = self.get_csf_results()

        rho_gt_list = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]

        # We will slowly change this to a PlotlyJS code with the right names!

        for reference_contrast_index in range(len(reference_contrast_list)):

            reference_contrast_value = reference_contrast_list[reference_contrast_index]

            test_contrast_list = conditions[f'ref_contrast_{reference_contrast_value}_test_contrast_list']
            human_gt_test_contrast_list =  human_result_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average']

            x1 = "["
            for i, el in enumerate(rho_test_list):
                if i>0:
                    x1 += ", "
                x1 += str(el)
            x1 += "]"

            y1 = "["
            for i, el in enumerate(test_contrast_list):
                if i>0:
                    y1 += ", "
                y1 += str(el)
            y1 += "]"

            x2 = "["
            for i, el in enumerate(rho_gt_list):
                if i>0:
                    x2 += ", "
                x2 += str(el)
            x2 += "]"

            y2 = "["
            for i, el in enumerate(human_gt_test_contrast_list):
                if i>0:
                    y2 += ", "
                y2 += str(el)
            y2 += "]"

            # Show the models results

            js_command += "\nvar ModelLineData"+str(reference_contrast_index)+" = {\n"

            js_command += "x: " + x1 + ",\ny: " + y1

            extra_params = "mode: 'lines',\ntype: 'scatter',\nline:{color: '"+self.colors[reference_contrast_index]+"', width:2},marker:{symbol:'circle-open', size:8},\nshowlegend: false,\nname: 'Model Prediction'"

            js_command += ",\n" + extra_params

            js_command += "\n};"

            # Show the ground truth results

            js_command += "\nvar GtLineData"+str(reference_contrast_index)+" = {\n"

            js_command += "x: " + x2 + ",\ny: " + y2

            extra_params = "mode: 'lines+markers',\ntype: 'scatter',\nline:{color: '"+self.colors[reference_contrast_index]+"', width:2, dash: 'dash'},marker:{symbol:'circle', size:8},\nshowlegend: false,\nname: 'HumanResults'"

            js_command += ",\n" + extra_params

            js_command += "\n};"
        
        # Add dummy data

        js_command += "\nvar ModelLineDataDummy = {\n"

        js_command += "x: [null],\ny: [null]"

        extra_params = "mode: 'lines',\ntype: 'scatter',\nline:{color: 'black', width:2},marker:{symbol:'circle-open', size:8},\nshowlegend: true,\nname: 'Model Prediction'"

        js_command += ",\n" + extra_params

        js_command += "\n};"


        js_command += "\nvar GtLineDataDummy = {\n"

        js_command += "x: [null],\ny: [null]"

        extra_params = "mode: 'lines+markers',\ntype: 'scatter',\nline:{color: 'black', width:2, dash: 'dash'},marker:{symbol:'circle', size:8},\nshowlegend: true,\nname: 'HumanResults'"

        js_command += ",\n" + extra_params

        js_command += "\n};"

        js_command += "\nvar data = [ModelLineData0, ModelLineData1, ModelLineData2, ModelLineData3, ModelLineData4, ModelLineData5, ModelLineData6, ModelLineData7, ModelLineDataDummy, GtLineData0, GtLineData1, GtLineData2, GtLineData3, GtLineData4, GtLineData5, GtLineData6, GtLineData7, GtLineDataDummy]"

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

        axes_info = "xaxis: {type:'log', showgrid: true, range: ["+str(log10(np.min(rho_gt_list)))+","+str(log10(np.max(rho_gt_list)))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'log', showgrid: true, range:["+str(log10(np.min(y_tick_vals)))+","+str(log10(np.max(y_tick_vals)))+"], tickvals: "+y+", ticktext: "+y+", title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"
        legend_info = "legend:{x:0.58, y:0.01, bgcolor: 'rgba(255, 255, 255, 0.5)', bordercolor: 'black', borderwidth: 1}"


        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+",\n"+legend_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def get_preview_folder(self):
        return self._preview_folder

    def short_name(self):
        return 'Contrast Constancy - Contrast Matching'
    
    def get_csf_results(self):
        path = os.path.join(self.csf_data_folder, 'contrast_constancy_sin_5_cpd.json')
        with open(path, 'r') as fp:
            csf_results = json.load(fp)
        
        return csf_results

    def get_ticks(self):
        x = [0.25, 0.5, 1, 2, 4, 8, 16]
        y = [0.001, 0.01, 0.1, 1]

        return x, y

    def get_row_header(self):
        return ['Test Contrast', 'Test Spatial Frequency (cpd)']

    def get_rows_conditions(self):
        return ['contrast_mask_matrix', 'contrast_test_matrix']

    def size(self):
        return 5, 5
    
    def units(self):
        return ['log', 'log'], [None, 'cpd']

    def preview_filepath(self):
        return "Contrast_Matching_supplementary.png"



    
    




    