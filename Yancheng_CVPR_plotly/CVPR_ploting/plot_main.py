import numpy as np
import torch
import os
from plot_zoo import *

# ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
#  'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
#  'contrast_detection_luminance', 'contrast_detection_area',
#  'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
#  'contrast_matching_cos_scale', 'contrast_matching_cos_scale_solve', 'contrast_matching_rafal_cos']

plot_tests_list = ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach',
                   'contrast_detection_SpF_Gabor_RG', 'contrast_detection_SpF_Gabor_YV',
                   'contrast_detection_luminance', 'contrast_detection_area',
                   'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking', 'contrast_matching_cos_scale_solve']
# plot_tests_list = ['contrast_matching_cos_scale_solve']

plt_class = plot_yancheng()
plt_class.plot_all_models(plot_tests_list)

