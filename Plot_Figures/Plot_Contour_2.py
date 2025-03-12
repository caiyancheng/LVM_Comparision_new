import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


model_dir_name_list = ['baseline', 'cvvdp', 'dinov2', 'sam2', 'vae', 'openclip']
sub_model_list_1 = [None, None, 'dinov2_vitl14', 'ppd_60/sam2.1_hiera_large', 'stable-diffusion-xl-base-1.0', 'ViT-B-16 - laion2b_s34b_b88k']
sub_model_list_2 = [None, None, 'dinov2_vitl14', 'sam2.1_hiera_large', 'stabilityai/stable-diffusion-xl-base-1.0', 'ViT-B-16 - laion2b_s34b_b88k']
sub_model_list_3 = [None, None, 'dinov2_vitl14', 'sam2.1_hiera_large', 'stabilityai/stable-diffusion-xl-base-1.0', 'ViT-B-16 - laion2b_s34b_b88k']
file_name_list_1 = ['cos similarity - baseline.png', 'JOD Score - cvvdp.png', 'cos similarity - dinov2.png', 'cos similarity - sam2 - ppd_60.png', 'cos similarity - vae.png', 'cos similarity - openclip.png']
file_name_list_2 = ['cos similarity - baseline - fix_random_seed.png', 'JOD Score - cvvdp - fix_random_seed.png', 'cos similarity - dinov2 fix_random_seed.png',
                    'cos similarity - sam2 fix_random_seed.png', 'cos similarity - vae fix_random_seed.png', 'cos similarity - openclip fix_random_seed.png']
file_name_list_3 = ['cos similarity - baseline.png', 'JOD Score - cvvdp.png', 'cos similarity - dinov2.png', 'cos similarity - sam2.png', 'cos similarity - vae.png', 'cos similarity - openclip.png']
column_dir_name_list = ['rho_YV', 'rho', 'c', 'c']

model_name_plot_list = ['No Encoder', 'ColorVideoVDP', 'DINOv2', 'SAM-2', 'VAE(SD)', 'OpenCLIP']
model_name_plot_list_2 = [None, None, 'ViT-L/14', 'SAM2.1-hiera-L', 'SD-xl-base-1.0', 'ViT-B/16-Laion']
# column_name_plot_list = ['Spatial Frequency (YV)', 'Band-limited Noise', 'Phase-Coherent Masking', 'Phase-Incoherent Masking']
column_name_plot_list = ['Spatial Frequency (YV)', 'Spatial Frequency (Noise)', 'Phase-Coherent Masking', 'Phase-Incoherent Masking']

gabor_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\plot\new_contour_plots/'
noise_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Band_lim_noise_test\plot\new_contour_plots/'
contrast_masking_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test\plot\new_contour_plots/'
contrast_masking_test_gabor_on_noise_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_masking_test_gabor_on_noise\plot\new_contour_plots/'
root_path_list = [gabor_test_root_path, noise_test_root_path, contrast_masking_test_root_path, contrast_masking_test_gabor_on_noise_root_path]

# 读取所有的24张png图片
# images = [Image.open(f'image_{i+1}.png') for i in range(24)]

# 创建6行4列的画布
fig, axes = plt.subplots(6, 4, figsize=(16, 15), dpi=300)

# 填充图片到图表中
for row in range(6): #每行
    for col in range(4): #每列
        ax = axes[row, col]
        if col == 0:
            file_name_list = file_name_list_1
            sub_model_list = sub_model_list_1
        elif col == 1:
            file_name_list = file_name_list_2
            sub_model_list = sub_model_list_2
        else:
            file_name_list = file_name_list_3
            sub_model_list = sub_model_list_3
        if row < 2:
            image_path = os.path.join(root_path_list[col], model_dir_name_list[row],
                                      f'different_{column_dir_name_list[col]}', file_name_list[row])
        else:
            image_path = os.path.join(root_path_list[col], model_dir_name_list[row],
                                      f'different_{column_dir_name_list[col]}', sub_model_list[row],
                                      file_name_list[row])
        img = Image.open(image_path)
        ax.imshow(np.asarray(img))
        ax.axis('off')

# 为每行添加模型名称
for row in range(6):
    ax = axes[row, 0]  # 第一列
    ax.text(-0.14, 0.5, model_name_plot_list[row], transform=ax.transAxes, fontsize=16,
            verticalalignment='center', rotation=90, fontweight='bold')
    if row >= 2:
        ax.text(-0.07, 0.5, model_name_plot_list_2[row], transform=ax.transAxes, fontsize=12,
            verticalalignment='center', rotation=90, fontweight='bold')

# 为每列添加列名称
for col in range(4):
    ax = axes[0, col]  # 第一行
    ax.set_title(column_name_plot_list[col], fontsize=16, fontweight='bold')

plt.subplots_adjust(wspace=-0.1, hspace=-0.08, left=0.02, right=1.01, top=0.98, bottom=0)
# 保存图像
plt.savefig('contour_plot_2.svg', dpi=300)
# plt.show()
