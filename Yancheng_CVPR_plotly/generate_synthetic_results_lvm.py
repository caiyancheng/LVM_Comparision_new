import pandas as pd
import logging
from tqdm import trange
import shutil

from pyst_lvm.ContrastDetection import *
from pyst_lvm.ContrastMasking import *
from pyst_lvm.ContrastMatching import *

from test_reports_utils.html_report import html_report
from test_reports_utils.js_report import js_report
import json

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)


    tests = [ContrastDetectionSpatialFreqAchGaborTest, ContrastDetectionSpatialFreqAchNoiseTest, 
             ContrastDetectionSpatialFreqRGGaborTest, ContrastDetectionSpatialFreqYVGaborTest,
             ContrastDetectionLuminanceTest, ContrastDetectionAreaTest, ContrastMaskingPhaseCoherentTest,
             ContrastMaskingPhaseIncoherentTest, ContrastMatchingTest]


    input_folder = 'CVPR_ploting/data_logs'
    output_folder = 'synthetic_test_webpage_lvm'

    generate_stimuli = False
    generate_metrics_results = True

    metrics = ['no_encoder', 'cvvdp_hdr', 'dino', 'dinov2', 'openclip', 'sam_float', 'sam2_float', 'mae', 'vae']
    metrics_names = ['No Encoder', 'ColorVideoVDP', 'DINO', 'DINOv2', 'OpenCLIP', 'SAM', 'SAM-2', 'MAE', 'SD-VAE']

    submetrics_names = [['/'], ['/'], ['ResNet-50', 'ViT-S/16', 'ViT-S/8', 'ViT-B/16', 'ViT-B/8', 'Xcit-S-12/16', 'Xcit-S-12/8',
                           'Xcit-M-24/16', 'Xcit-M-24/8'], 
                           ['ViT-S/14', 'ViT-B/14', 'ViT-L/14', 'ViT-g/14', 'ViT-S/14 + reg', 'ViT-B/14 + reg', 
                            'ViT-L/14 + reg', 'ViT-G/14 + reg'], 
                            ['ResNet-50', 'ResNet-50', 'ResNet-101', 'ResNet-101', 'ConvNext-B-w', 'ConvNext-B-w',
                             'ConvNext-L-d', 'ConvNext-XXL', 'ViT-B/32', 'ViT-B/32', 'ViT-B/16', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14'],
                             ['ViT-B-SAM', 'ViT-L-SAM', 'ViT-H-SAM'], ['SAM2.1-hiera-tiny', 'SAM2.1-hiera-S', 'SAM2.1-hiera-B+', 'SAM2.1-hiera-L'],
                             ['ViT-B-MAE', 'ViT-L-MAE', 'ViT-H-MAE'], ['SD-v1-5', 'SD-xl-base-1.0']]
    
    submetrics_datasets = [['/'], ['/'], ['ImageNet', 'ImageNet', 'ImageNet', 'ImageNet', 'ImageNet', 'ImageNet', 'ImageNet', 'ImageNet', 'ImageNet'],
                           ['LVD-142M', 'LVD-142M', 'LVD-142M', 'LVD-142M', 'LVD-142M', 'LVD-142M', 'LVD-142M', 'LVD-142M'],
                           ['OpenAI', 'YFCC-15M', 'OpenAI', 'YFCC-15M', 'LAION-2B', 'LAION-2B+', 'LAION-2B+', 'LAION-2B+', 
                            'OpenAI', 'LAION-2B', 'OpenAI', 'LAION-2B', 'OpenAI', 'LAION-2B'],
                            ['SA-1B', 'SA-1B', 'SA-1B'], ['SA-V', 'SA-V', 'SA-V', 'SA-V'], ['ImageNet', 'ImageNet', 'ImageNet'], ['LAION', 'LAION']]
    
    submetricss = [["no_encoder"], ["cvvdp_hdr"], ["dino_resnet50", "dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8", "dino_xcit_small_12_p16", "dino_xcit_small_12_p8", "dino_xcit_medium_24_p16", "dino_xcit_medium_24_p8"],
                  ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"],
                  ["RN50_openai", "RN50_yfcc15m", "RN101_openai", "RN101_yfcc15m", 
                   "convnext_base_w_laion2b_s13b_b82k", "convnext_base_w_laion2b_s13b_b82k_augreg", "convnext_large_d_laion2b_s26b_b102k_augreg", 
                   "convnext_xxlarge_laion2b_s34b_b82k_augreg", "ViT-B-32_openai", "ViT-B-32_laion2b_s34b_b79k", 
                   "ViT-B-16_openai", "ViT-B-16_laion2b_s34b_b88k", "ViT-L-14_openai", "ViT-L-14_laion2b_s32b_b82k"],
                   ["sam_vit_b_01ec64", "sam_vit_l_0b3195", "sam_vit_h_4b8939"], ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"],
                   ["vit-mae-base", "vit-mae-large", "vit-mae-huge"], ["stable-diffusion-v1-5", "stable-diffusion-xl-base-1.0"]]

    # Create the directory and its associated html file!

    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder, 'html_files'))
        os.makedirs(os.path.join(output_folder, 'previews'))
        os.makedirs(os.path.join(output_folder, 'css_files'))
        os.makedirs(os.path.join(output_folder, 'js_files'))

    shutil.copyfile(os.path.join('test_reports_utils', 'style.css'), os.path.join(output_folder, 'css_files', 'style.css'))
    shutil.copyfile(os.path.join('test_reports_utils', 'plotly-2.35.2.min.js'), os.path.join(output_folder, 'js_files', 'plotly-2.35.2.min.js'))

    header_file = os.path.join('test_reports_utils', "header_plotly.html")
    hr_main = html_report(os.path.join(output_folder, 'index.html'), header_file=header_file )
    hr_main.insert_file(os.path.join('test_reports_utils', "nav.html"))
    hr_main.beg_div(class_tag='other_content')
    hr_main.insert_text('Please select one of the tests in the sidebar to view the models performance in the test.')
    hr_main.close_div()

    # Get metrics alignments scores
    scores_path = "CVPR_ploting/Score_Results_2_new.csv"
    scores_df = pd.read_csv(scores_path)

    

    for T in tests:
        Test = T()
        test_name = Test.short_name()

        [test_rows, test_columns] = Test.size()
        [x_variable_name, y_variable_name] = Test.get_row_header()

        logging.info(f"Generating results for {test_name}")

        results_folder = os.path.join(input_folder, Test.get_preview_folder())

        # Here we want to create an html file for each test! The html file will contain the results plots of its own test!

        test_html_filename = test_name.lower().replace(' - ', '_').replace(' (', '_').replace(')', '').replace(' ', '_').replace('-', '_')+'.html'
        hr_test = html_report(os.path.join(output_folder, test_html_filename), header_file=header_file)

        js_filename = 'script_'+test_name.lower().replace(' - ', '_').replace(' (', '_').replace(')', '').replace(' ', '_').replace('-', '_')+'.js'
        js_main = js_report(os.path.join(output_folder, 'js_files', js_filename))

        hr_test.insert_file(os.path.join('test_reports_utils', "nav.html"))


        hr_test.beg_div(class_tag='other_content')
        #hr_test.insert_separator()
        hr_test.insert_header(test_name)

        hr_test.insert_text('The plots are generated with Plotly JS, please wait for few seconds for all the plots to be generated.')
        hr_test.insert_text('The countour plots are dynamic, you can hover over them and read the results, or zoom in and zoom out.')
        
        if test_name != 'Contrast Constancy - Contrast Matching':
            hr_test.insert_text('Each contour plot corresponds to the results of a model for the "{the_test_name}" test in terms of the latent space dissimilarity S<sub>ac</sub>, while the red line corresponds to the ground truth. The test stimuli examples can be found <a href={link}>here</a>.'.format(link=os.path.join("previews", Test.preview_filepath()), the_test_name=test_name))
        else:
            hr_test.insert_text('Each plot corresponds to the results from the "Contrast Matching" experiment, where different colors represent different contrast C<sub>r</sub> values. The dashed lines are human results, while the solid lines are model-predicted results. The test stimuli examples can be found <a href={link}>here.'.format(link=os.path.join("previews", Test.preview_filepath())))

        # Now, we want to add all metric results to this test webpage

        hr_test.beg_div(class_tag='image-container')

        for metric, metric_name, submetrics, submetrics_name, submetrics_dataset in zip(metrics, metrics_names, submetricss, submetrics_names, submetrics_datasets):
            if metric_name == 'SD-VAE':
                X = 1
            try:

                metric_predictions_file = os.path.join(results_folder, metric, metric+'_'+Test.get_preview_folder()+'.json')

                with open(metric_predictions_file, 'r') as fp:
                    metric_results = json.load(fp)

                for submetric, submetric_name, submetric_dataset in zip(submetrics, submetrics_name, submetrics_dataset):

                    score_name = 'Alignment score'
                    if test_name == 'Contrast Constancy - Contrast Matching':
                        score_name = 'RMSE'

                    if len(submetrics)>1:
                        score = scores_df[(scores_df['Models'] == metric_name) & (scores_df['Architecture'] == submetric_name) & (scores_df['Training dataset'] == submetric_dataset)][test_name].to_numpy()[0]
                        final_metric_name = metric_name +' ['+submetric_name+' - '+submetric_dataset+']<br>'+score_name+' = '+str(score)
                        metric_name_id = metric_name+"_"+submetric.replace('-', '_')
                        
                    else:
                        score = scores_df[scores_df['Models'] == metric_name][test_name].to_numpy()[0]
                        final_metric_name = metric_name+'<br>'+score_name+' = '+str(score)
                        metric_name_id = metric_name
                        

                    metric_id = test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')+'_'+metric_name_id.lower().replace(' ', '_')

                    if test_name != 'Contrast Constancy - Contrast Matching':
                        x_condition = metric_results[submetric][Test.get_rows_conditions()[0]]
                        y_condition = metric_results[submetric][Test.get_rows_conditions()[1]]

                        if metric.startswith('cvvdp'):
                            predictions = metric_results[submetric]['JOD_scale_matrix']
                        else:
                            predictions = metric_results[submetric]['arccos_cos_similarity_matrix']

                        if generate_metrics_results:
                            js_command = Test.plotly(x_condition, y_condition, predictions, title=final_metric_name, fig_id=metric_id)

                    else:
                        # We need a special code for contrast matching as it is widely different compared to what we have already!

                        conditions = metric_results[submetric]
                        reference_contrast_list = metric_results['reference_contrast_list']
                        rho_test_list = metric_results['rho_test_list']

                        if generate_metrics_results:
                            js_command = Test.plotly(conditions, reference_contrast_list, rho_test_list, title=final_metric_name, fig_id=metric_id)

                    # Similarly to the main html file we need to create a JS file!

                    js_main.println(js_command)

                    hr_test.beg_div(id_tag = metric_id)
                    hr_test.close_div()

            except:
                logging.error(f'Failed on test {test_name} and the {metric_name} metric')

        hr_test.close_div()
        hr_test.close_div()

        hr_test.close(js_filename)
        js_main.close()

    hr_main.close()
    





