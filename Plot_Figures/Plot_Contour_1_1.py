import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


model_dir_name_list = ['dinov2', 'sam2', 'vae', 'openclip']
sub_model_list = ['dinov2_vitl14', 'ppd_60/sam2.1_hiera_large', 'stable-diffusion-xl-base-1.0', 'ViT-B-16 - laion2b_s34b_b88k']
file_name_list = ['cos similarity - dinov2.png', 'cos similarity - sam2 - ppd_60.png', 'cos similarity - vae.png', 'cos similarity - openclip.png']
column_dir_name_list = ['area', 'luminance', 'rho', 'rho_RG']

model_name_plot_list = ['DINOv2', 'SAM-2', 'VAE(SD)', 'OpenCLIP']
model_name_plot_list_2 = ['ViT-L/14', 'SAM2.1-hiera-L', 'SD-xl-base-1.0', 'ViT-B/16-Laion']
column_name_plot_list = ['Area', 'Luminance', 'Spatial Frequency', 'Spatial Frequency (RG)']

gabor_test_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Gabor_test\plot\new_contour_plots/'

# 读取所有的24张png图片
# images = [Image.open(f'image_{i+1}.png') for i in range(24)]

# 创建6行4列的画布
fig, axes = plt.subplots(len(model_dir_name_list), len(column_dir_name_list), figsize=(16, 10), dpi=300)

# 填充图片到图表中
for row in range(len(model_dir_name_list)): #每行
    for col in range(len(column_dir_name_list)): #每列
        ax = axes[row, col]
        if row < 0:
            image_path = os.path.join(gabor_test_root_path, model_dir_name_list[row],
                                      f'different_{column_dir_name_list[col]}', file_name_list[row])
        else:
            image_path = os.path.join(gabor_test_root_path, model_dir_name_list[row],
                                      f'different_{column_dir_name_list[col]}', sub_model_list[row],
                                      file_name_list[row])
        img = Image.open(image_path)
        ax.imshow(np.asarray(img))
        ax.axis('off')

# 为每行添加模型名称
for row in range(len(model_dir_name_list)):
    ax = axes[row, 0]  # 第一列
    ax.text(-0.14, 0.5, model_name_plot_list[row], transform=ax.transAxes, fontsize=16,
            verticalalignment='center', rotation=90, fontweight='bold')
    if row >= 0:
        ax.text(-0.07, 0.5, model_name_plot_list_2[row], transform=ax.transAxes, fontsize=12,
            verticalalignment='center', rotation=90, fontweight='bold')

# 为每列添加列名称
for col in range(len(column_dir_name_list)):
    ax = axes[0, col]  # 第一行
    ax.set_title(column_name_plot_list[col], fontsize=16, fontweight='bold')

plt.subplots_adjust(wspace=-0.05, hspace=-0.08, left=0.02, right=1, top=0.98, bottom=0)
# 保存图像
# plt.savefig('contour_plot_1.svg', dpi=300)
plt.show()
