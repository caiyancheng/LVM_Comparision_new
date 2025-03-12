import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

model_dir_name_list = ['dinov2', 'sam2', 'vae', 'openclip']
sub_model_list = ['dinov2_vitl14', 'sam2.1_hiera_large', 'stable-diffusion-xl-base-1.0', 'ViT-B-16 - laion2b_s34b_b88k']
file_name_list = ['cos similarity - dinov2.png', 'cos similarity - sam2.png', 'cos similarity - vae.png', 'cos similarity - openclip.png']
file_name_f_list = ['cos similarity - dinov2 - fix_random_seed.png', 'cos similarity - sam2 - fix_random_seed.png', 'cos similarity - vae - fix_random_seed.png', 'cos similarity - openclip - fix_random_seed.png']
file_name_cm_list = ['cosine_sim_contrast_matching.png', 'cosine_sim_contrast_matching.png', 'cosine_sim_contrast_matching.png', 'cosine_sim_contrast_matching.png']
row_dir_name_list = ['area', 'luminance', 'rho', 'rho_RG', 'rho_YV', 'rho', 'c', 'c', 'rho']

Plot_Column_model_name_list = ['DINOv2', 'SAM-2', 'VAE(SD)', 'OpenCLIP']
Plot_Column_submodel_name_list = ['ViT-L/14', 'SAM2.1-hiera-L', 'SD-xl-base-1.0', 'ViT-B/16-Laion']
Plot_Row_name_list = ['Area', 'Luminance', 'Spatial Frequency', 'Spatial Frequency', 'Spatial Frequency', 'Spatial Frequency', 'Phase-Coherent', 'Phase-Incoherent', 'Contrast']
Plot_Row_name_list_2 = ['Gabor (Ach.)', 'Gabor (Ach.)', 'Gabor (Ach.)', 'Gabor (RG)', 'Gabor (YV)', 'Noise (Ach.)', 'Masking', 'Masking', 'Matching']

gabor_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\plot\new_contour_plots/'
noise_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Band_lim_noise_test\plot\new_contour_plots/'
contrast_masking_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test\plot\new_contour_plots/'
contrast_masking_test_gabor_on_noise_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test_gabor_on_noise\plot\new_contour_plots/'
contrast_matching_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy\plot\new_contour_plots/'

row_root_path_list = [gabor_test_root_path, gabor_test_root_path, gabor_test_root_path, gabor_test_root_path, gabor_test_root_path,
                      noise_test_root_path, contrast_masking_test_root_path, contrast_masking_test_gabor_on_noise_root_path, contrast_matching_root_path]
fig, axes = plt.subplots(len(Plot_Row_name_list), len(Plot_Column_model_name_list), figsize=(17, 20.5)) #, dpi=300)

# 填充图片到图表中
for row in range(len(Plot_Row_name_list)): #每行
    if row == 5:
        file_name_l = file_name_f_list
    elif row == len(Plot_Row_name_list) - 1:
        file_name_l = file_name_cm_list
    else:
        file_name_l = file_name_list
    for col in range(len(Plot_Column_model_name_list)): #每列
        ax = axes[row, col]
        root_path = row_root_path_list[row]
        model_dir = model_dir_name_list[col]
        sub_model_dir = sub_model_list[col]
        file_name = file_name_l[col]
        image_path = os.path.join(root_path, model_dir, f'different_{row_dir_name_list[row]}', sub_model_dir, file_name)
        img = Image.open(image_path)
        ax.imshow(np.asarray(img))
        ax.axis('off')

# 为每行添加测试名称
for row in range(len(Plot_Row_name_list)):
    ax = axes[row, 0]  # 第一列
    ax.text(-0.16, 0.5, Plot_Row_name_list[row], transform=ax.transAxes, fontsize=14,
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

plt.subplots_adjust(wspace=0.02, hspace=0, left=0.05, right=0.999, top=0.975, bottom=0)
# 保存图像
plt.savefig('contour_plot_3.svg', dpi=300)
# plt.show()