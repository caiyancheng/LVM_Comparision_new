import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode

Luminance_min = 1e-4
display_encode_tool = display_encode(400)

def create_cycdeg_image(im_size, pix_per_deg):
    nyquist_freq = 0.5 * pix_per_deg
    KX0 = (np.mod(1 / 2 + np.arange(im_size[1]) / im_size[1], 1) - 1 / 2)
    KX1 = KX0 * nyquist_freq * 2
    KY0 = (np.mod(1 / 2 + np.arange(im_size[0]) / im_size[0], 1) - 1 / 2)
    KY1 = KY0 * nyquist_freq * 2
    XX, YY = np.meshgrid(KX1, KY1)
    D = np.sqrt(XX ** 2 + YY ** 2)
    return D

def generate_masking_stimulus_band_limit_noise(W, H, T_freq_band, R_freq_band, L_b, contrast_mask, contrast_test, ppd):
    np.random.seed(8)
    Noise = np.random.randn(W, H)
    T_Noise_f = np.fft.fft2(Noise)
    R_Noise_f = np.fft.fft2(Noise)
    rho = create_cycdeg_image([W, H], ppd)
    T_log2_freq_band = np.log2(T_freq_band)
    R_log2_freq_band = np.log2(R_freq_band)
    T_freq_edge_low = 2 ** (T_log2_freq_band - 0.5)
    T_freq_edge_high = 2 ** (T_log2_freq_band + 0.5)
    R_freq_edge_low = 2 ** (R_log2_freq_band - 0.5)
    R_freq_edge_high = 2 ** (R_log2_freq_band + 0.5)
    T_Noise_f[(rho < T_freq_edge_low) | (rho > T_freq_edge_high)] = 0
    R_Noise_f[(rho < R_freq_edge_low) | (rho > R_freq_edge_high)] = 0
    T_Noise_bp = np.real(np.fft.ifft2(T_Noise_f))
    R_Noise_bp = np.real(np.fft.ifft2(R_Noise_f))
    T_Noise_bp = T_Noise_bp / np.std(T_Noise_bp)
    R_Noise_bp = R_Noise_bp / np.std(R_Noise_bp)
    size_deg = np.array([W, H]) / ppd
    sigma = 2 / 4 #T_freq_band
    XX, YY = np.meshgrid(np.linspace(0, size_deg[0], W),
                         np.linspace(0, size_deg[1], H))
    gauss_env = np.exp(-((XX - size_deg[0] / 2) ** 2 + (YY - size_deg[1] / 2) ** 2) / (2 * sigma ** 2))
    img_mask = L_b + R_Noise_bp * L_b * contrast_mask
    img_target = T_Noise_bp * L_b * contrast_test * gauss_env
    S = np.maximum(Luminance_min, img_mask + img_target)
    return S

def generate_contrast_masking_band_limit_noise(W, H, T_freq_band, R_freq_band, L_b, contrast_mask, contrast_test, ppd):
    T_vid = generate_masking_stimulus_band_limit_noise(W, H, T_freq_band, R_freq_band, L_b, contrast_mask, contrast_test, ppd)
    R_vid = generate_masking_stimulus_band_limit_noise(W, H, T_freq_band, R_freq_band, L_b, contrast_mask, 0, ppd)
    T_vid[T_vid < 0] = 0
    R_vid[R_vid < 0] = 0
    return T_vid, R_vid

def plot_contrast_masking_band_limit_noise(T_vid, R_vid):
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
    # R_vid_c = display_encode_tool.L2C(R_vid) * 255
    plt.figure(figsize=(4, 4))
    plt.imshow(T_vid_c, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(
        f'contrast_mask = {contrast_mask}, contrast_test = {contrast_test}, \n'
        f'T_freq_b = {T_freq_band} cpd, R_freq_b = {R_freq_band} cpd, \n'
        f'L_b = {L_b} $cd/m^2$,  ppd = {ppd},\n'
        f' W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    # 示例参数
    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    T_freq_band = 4 / scale_k1 / scale_k2
    R_freq_band = 4 / scale_k1 / scale_k2
    L_b = 100  # Luminance of the background
    contrast_mask = 0.1
    contrast_test = 0.5
    ppd = 60 / scale_k1

    T_vid, R_vid = generate_contrast_masking_band_limit_noise(W, H, T_freq_band, R_freq_band, L_b, contrast_mask, contrast_test, ppd)
    plot_contrast_masking_band_limit_noise(T_vid, R_vid)

