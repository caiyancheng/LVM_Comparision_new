import matplotlib.pyplot as plt
import json
import os
import numpy as np

levels_list = [1, 2, 3, 4, 5]
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg'] #, 'dinov2_vitg14_reg']
noise_result_root_path = r'E:\Py_codes\LVM_Comparision\Noise_Generalization_Test\noise_test_results\dinov2\gaussian_noise'

for backbone_name in all_backbone_list:
    result_json_path = os.path.join(noise_result_root_path, backbone_name, 'cos_sim_result.json')
    with open(result_json_path, 'r') as fp:







