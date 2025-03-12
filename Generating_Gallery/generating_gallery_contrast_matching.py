import json

from Sinusoidal_grating_generator.generate_plot_sinusoidal_grating import generate_sinusoidal_grating
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from display_encoding import display_encode
display_encode_tool = display_encode(400)

W = 224
H = 224
rho_reference = 5
rho_test_list_gt = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
L_b = 10
O = 0
ppd = 60

with open(r'../Feature_Similarity_paper_report_auto/gt_json/contrast_constancy_sin_5_cpd.json', 'r') as fp:
    json_data = json.load(fp)
reference_contrast_list = json_data['average_reference_contrast'] #length-8

fig = plt.figure(figsize=(16, 12.8), dpi=300)
gs = gridspec.GridSpec(len(reference_contrast_list), 2 + len(rho_test_list_gt), width_ratios=[1, 0.3] + [1] * len(rho_test_list_gt))
# fig, axes = plt.subplots(len(reference_contrast_list), 1 + len(rho_test_list_gt), figsize=(22, 11.4), dpi=300)
plt.subplots_adjust(left=0.05, right=0.999, top=0.92, bottom=0.001, wspace=0.04, hspace=0.04)

for reference_contrast_index in range(len(reference_contrast_list)):
    reference_contrast_value = reference_contrast_list[reference_contrast_index]
    Human_test_contrast_list = json_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average']

    T_vid = generate_sinusoidal_grating(W=W, H=H,
                                        spatial_frequency=rho_reference,
                                        orientation=O, L_b=L_b,
                                        contrast=reference_contrast_value, ppd=ppd)
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    T_vid_c = np.array(T_vid_c).astype(np.uint8)
    ax = fig.add_subplot(gs[reference_contrast_index, 0])
    ax.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255)
    ax.text(-0.08, 0.5, f'{reference_contrast_value:.3f}',
            va='center', ha='center', rotation=90, fontsize=16,
            transform=ax.transAxes)
    ax.axis('off')

    for rho_index in range(len(rho_test_list_gt)):
        rho_value = rho_test_list_gt[rho_index]
        test_contrast = Human_test_contrast_list[rho_index]
        T_vid = generate_sinusoidal_grating(W=W, H=H,
                                            spatial_frequency=rho_value,
                                            orientation=O, L_b=L_b,
                                            contrast=test_contrast, ppd=ppd)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        T_vid_c = np.array(T_vid_c).astype(np.uint8)
        ax = fig.add_subplot(gs[reference_contrast_index, 2 + rho_index])
        if reference_contrast_index == 0:
            ax.set_title(f'{rho_value:.1f}', fontsize=16)
        ax.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')

fig.text(0.09, 0.93, "Reference", ha='center', fontsize=20)
fig.text(0.01, 0.5, "Reference Contrast", va='center', ha='center', rotation='vertical', fontsize=20)
fig.text(0.6, 0.975, 'Test --- Spatial Frequency (cpd)', ha='center', fontsize=20)

plt.annotate('', xy=(0.997, 0.955), xytext=(0.2, 0.955),
             arrowprops=dict(facecolor='black', shrink=0.05),
             xycoords='figure fraction')
plt.annotate('', xy=(0.025, 0.92), xytext=(0.025, 0.003),
             arrowprops=dict(facecolor='black', shrink=0.05),
             xycoords='figure fraction')
# plt.show()
# plt.savefig('Real_sup/Contrast_Matching_supplementary.svg', dpi=300)
plt.savefig('Sup_pngs/Contrast_Matching_supplementary.png', dpi=300)
plt.close()
