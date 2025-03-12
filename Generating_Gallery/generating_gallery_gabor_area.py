from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
from Contrast_masking_generator.generate_plot_contrast_masking_band_limit_noise import generate_contrast_masking_band_limit_noise
import numpy as np
import matplotlib.pyplot as plt
import math
from display_encoding import display_encode
display_encode_tool = display_encode(400)

size_of_gallery = [5,5]

##Area:
default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), size_of_gallery[1]) #每列是不同的Area
R_list = (Area_list / math.pi) ** 0.5
default_rho = 8
contrast_list = np.logspace(np.log10(0.1), np.log10(1), size_of_gallery[0]) #每行是不同的Contrast
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60

fig, axes = plt.subplots(size_of_gallery[0], size_of_gallery[1], figsize=(10, 10), dpi=300)
plt.subplots_adjust(left=0.08, right=0.999, top=0.92, bottom=0.001, wspace=0.04, hspace=0.04)
for contrast_index in range(len(contrast_list)):
    contrast_value = contrast_list[contrast_index]
    for R_index in range(len(R_list)):
        R_value = R_list[R_index]
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                            L_b=default_L_b, contrast=contrast_value, ppd=default_ppd,
                                            color_direction='ach')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        T_vid_c = np.array(T_vid_c).astype(np.uint8)
        ax = axes[contrast_index, R_index]
        ax.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        # ax.text(0.5, -0.02, f"Contrast: {contrast_value:.2f}\nRadius: {R_value:.2f}\u00B0",
        #         transform=ax.transAxes, ha='center', va='top', fontsize=12)
# 添加横轴（radius）和纵轴（contrast）的标签
for ax, R_value in zip(axes[0], R_list):
    ax.set_title(f'{R_value:.3f}\u00B0', fontsize=16)
#
for ax, contrast_value in zip(axes[:, 0], contrast_list):
    ax.text(-0.06, 0.5, f'{contrast_value:.3f}',
            va='center', ha='center', rotation=90, fontsize=16,
            transform=ax.transAxes)
    # ax.set_visible(True)

# 在左侧添加纵向箭头
fig.text(0.007, 0.5, 'Contrast', va='center', rotation='vertical', fontsize=20)
fig.text((0.08+0.997)/2, 0.975, 'Radius (visual degree)', ha='center', fontsize=20)

# 添加从左到右的箭头（横轴）
plt.annotate('', xy=(0.997, 0.955), xytext=(0.08, 0.955),
             arrowprops=dict(facecolor='black', shrink=0.05),
             xycoords='figure fraction')

# 添加从上到下的箭头（纵轴）
plt.annotate('', xy=(0.045, 0.003), xytext=(0.045, 0.92),
             arrowprops=dict(facecolor='black', shrink=0.05),
             xycoords='figure fraction')


# plt.tight_layout(pad=-0.5)
# plt.show()
plt.savefig('Gabor_Area_Contrast_sup.png', dpi=300)
plt.close()