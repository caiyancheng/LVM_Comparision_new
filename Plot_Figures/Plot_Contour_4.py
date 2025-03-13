import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

model_dir_name_list = ['dino', 'dinov2', 'sam2_float', 'vae', 'openclip']
sub_model_list = ['dino_vitb8', 'dinov2_vitg14', 'sam2.1_hiera_large', 'stable-diffusion-xl-base-1.0', 'ViT-L-14_laion2b_s32b_b82k']

Plot_Column_model_name_list = ['DINO', 'DINOv2', 'SAM-2', 'SD-VAE', 'OpenCLIP']
Plot_Column_submodel_name_list = ['ViT-B/8', 'ViT-G/14', 'SAM2.1-hiera-L', 'SD-xl-base-1.0', 'ViT-L/14-Laion']

Row_tests_list = ['contrast_detection_SpF_Gabor_ach', 'contrast_detection_SpF_Noise_ach', 'contrast_detection_SpF_Gabor_RG',
                  'contrast_detection_SpF_Gabor_YV', 'contrast_detection_luminance', 'contrast_detection_area',
                  'contrast_masking_phase_coherent_masking', 'contrast_masking_phase_incoherent_masking',
                  'contrast_matching_cos_scale_solve_no_scaler']
Plot_Row_name_list_1 = ['Spatial Frequency', 'Spatial Frequency', 'Spatial Frequency', 'Spatial Frequency',
                        'Luminance', 'Area', 'Phase-coherent', 'Phase-incoherent', 'Contrast']
Plot_Row_name_list_2 = ['Gabor Ach.', 'Noise Ach.', 'Gabor RG', 'Gabor YV', 'Gabor Ach.', 'Gabor Ach.', 'Masking', 'Masking', 'Matching']

plot_figs_root_path = r'E:\Py_codes\LVM_Comparision_new\Feature_Similarity_paper_report_auto\plot_picts'

fig, axes = plt.subplots(len(Row_tests_list), len(Plot_Column_model_name_list), figsize=(18.5, 20.5)) #, dpi=300)

symbols = [f"({chr(97 + i)})" for i in range(len(Row_tests_list))]
# 填充图片到图表中
for row in range(len(Row_tests_list)):
    for col in range(len(Plot_Column_model_name_list)): #每列
        ax = axes[row, col]
        Row_name = Row_tests_list[row]
        if Row_name.startswith('contrast_matching'):
            image_path = os.path.join(plot_figs_root_path, f'test_{Row_tests_list[row]}', model_dir_name_list[col],
                                      f'{sub_model_list[col]}-cos.png')
        else:
            image_path = os.path.join(plot_figs_root_path, f'test_{Row_tests_list[row]}', model_dir_name_list[col], f'{sub_model_list[col]}-arccos_cos_similarity_matrix.png')
        img = Image.open(image_path)
        ax.imshow(np.asarray(img))
        ax.axis('off')

# 为每行添加测试名称
for row in range(len(Row_tests_list)):
    ax = axes[row, 0]  # 第一列
    ax.text(-0.3, 0.5, symbols[row], transform=ax.transAxes, fontsize=14,  # 在左边添加符号
            verticalalignment='center', horizontalalignment='center', fontweight='bold')
    ax.text(-0.16, 0.5, Plot_Row_name_list_1[row], transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center', rotation=90, fontweight='bold')
    ax.text(-0.08, 0.5, Plot_Row_name_list_2[row], transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center', rotation=90, fontweight='bold')

# 为每列添加模型名称
for col in range(len(Plot_Column_model_name_list)):
    ax = axes[0, col]  # 第一行
    ax.text(0.5, 1.16, Plot_Column_model_name_list[col], transform=ax.transAxes, fontsize=16,
            verticalalignment='center', horizontalalignment='center', fontweight='bold')
    ax.text(0.5, 1.07, Plot_Column_submodel_name_list[col], transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center', fontweight='bold')
    # ax.set_title(Plot_Column_model_name_list[col], fontsize=16, fontweight='bold')

plt.subplots_adjust(wspace=0.01, hspace=0, left=0.06, right=0.999, top=0.975, bottom=0)
# 保存图像
plt.savefig('contour_plot_10.svg', dpi=250)
# plt.show()