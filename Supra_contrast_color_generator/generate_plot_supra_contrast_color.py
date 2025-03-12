import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode
from Color_space_Transform import *

Luminance_min = 1e-4
display_encode_tool = display_encode(400, 2.2)
dkl_ratios = np.array([1, 0.610649, 4.203636])
white_point_d65 = np.array([0.9505, 1.0000, 1.0888])

def generate_supra_contrast_color(H, W, rho, L_b, contrast_test, ppd, color_direction):
    if color_direction == 'ach':
        col_dir = np.array([dkl_ratios[0], 0, 0])
    elif color_direction == 'rg':
        col_dir = np.array([0, dkl_ratios[1], 0])
    elif color_direction == 'yv':
        col_dir = np.array([0, 0, dkl_ratios[2]])
    else:
        raise ValueError('Color Direction Value is not correct. We only support ach, rg, yv')
    snw = np.sin(np.linspace(0, 2 * np.pi * rho * W / ppd, W))
    sqw = (snw > 0).astype(float) * 2 - 1
    C_dkl = lms2dkl_d65(xyz2lms2006(white_point_d65 * L_b))
    I_dkl_ref = np.ones((H, W, 3)) * C_dkl.reshape(1, 1, 3)
    I_dkl_test = (I_dkl_ref + (sqw * contrast_test * L_b)[:, np.newaxis] *
                  np.ones((H, 1)) * col_dir.reshape(1, 1, 3))
    T_vid = cm_xyz2rgb(lms2006_2xyz(dkl2lms_d65(I_dkl_test)))
    R_vid = cm_xyz2rgb(lms2006_2xyz(dkl2lms_d65(I_dkl_ref)))
    assert np.all(T_vid >= 0), "We cannot have any out of gamut colours"

    return T_vid, R_vid

def plot_supra_contrast_color(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C(T_vid)
    R_vid_c = display_encode_tool.L2C(R_vid)
    plt.figure(figsize=(4, 4))
    plt.imshow(T_vid_c, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(
        f'contrast_test = {contrast_test}, color_dir = {color_direction}\n rho = {rho} cpd, L_b = {L_b} $cd/m^2$, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    # 示例参数
    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    rho = 1 / scale_k1 / scale_k2  # Spatial frequency of the Gabor stimulus (cycles per degree)
    L_b = 100  # Luminance of the background
    contrast_test = 0.2
    ppd = 60 / scale_k1
    color_direction = 'yv'

    T_vid, R_vid = generate_supra_contrast_color(H, W, rho, L_b, contrast_test, ppd, color_direction)
    plot_supra_contrast_color(T_vid, R_vid)

