import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import json

name = 'dino'
json_path_list = [rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\test\new_data_logs\{name}\different_area/{name}_test_on_gabors_different_area_contour_plot_ppd_60_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\test\new_data_logs\{name}\different_luminance/{name}_test_on_gabors_different_L_b_contour_plot_ppd_60_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\test\new_data_logs\{name}\different_rho/{name}_test_on_gabors_different_rho_contour_plot_ppd_60_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\test\new_data_logs\{name}\different_rho_RG/{name}_test_on_gabors_different_rho_contour_plot_ppd_60_RG_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\test\new_data_logs\{name}\different_rho_YV/{name}_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Band_lim_noise_test\test\new_data_logs\{name}\different_rho/{name}_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_60_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test\test\new_data_logs\{name}\different_c/{name}_test_on_contrast_masking_different_c_contour_plot_ppd_60_final.json',
                 rf'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test_gabor_on_noise\test\new_data_logs\{name}\different_c/{name}_test_on_contrast_masking_different_c_contour_plot_ppd_60_final.json']

with open(json_path_list[0], 'r') as fp:
    json_data = json.load(fp)
    backbone_list = json_data['backbone_name']

max_L1 = np.zeros(len(backbone_list))
max_L2 = np.zeros(len(backbone_list))
min_cos = np.ones(len(backbone_list))

for json_path in json_path_list:
    with open(json_path, 'r') as fp:
        json_data = json.load(fp)
        max_L1_here = np.max(np.array(json_data['final_feature_L1_similarity_matrix']), axis=(1, 2))
        max_L2_here = np.max(np.array(json_data['final_feature_L2_similarity_matrix']), axis=(1, 2))
        min_cos_here = np.min(np.array(json_data['final_feature_cos_similarity_matrix']), axis=(1, 2))
        max_L1 = np.maximum(max_L1_here, max_L1)
        max_L2 = np.maximum(max_L2_here, max_L2)
        min_cos = np.minimum(min_cos_here, min_cos)

json_data = {}
json_data['max_L1'] = max_L1.tolist()
json_data['max_L2'] = max_L2.tolist()
json_data['min_cos'] = min_cos.tolist()

with open(f'max_L1_L2_{name}.json', 'w') as fp:
    json.dump(json_data, fp)

