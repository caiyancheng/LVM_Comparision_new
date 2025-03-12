import numpy as np


def rgb_to_lms(rgb):
    # RGB to LMS transformation matrix
    M = np.array([
        [0.214808, 0.751035, 0.045156],
        [0.022882, 0.940534, 0.076827],
        [0.0000, 0.0165, 0.999989]
    ])

    # Normalize RGB values to [0, 1]
    rgb = np.array(rgb) / 255.0

    # Perform the matrix multiplication
    lms = M @ rgb

    return lms


def lms_to_rgb(lms):
    # LMS to RGB transformation matrix
    M_inv = np.array([
        [5.0883, -4.0645, 0.08250],
        [-0.1239, 1.1637, -0.08381],
        [0.00205, -0.01920, 1.00139]
    ])

    # Perform the matrix multiplication
    rgb = np.dot(M_inv, lms)

    # Clip values to [0, 1] and scale to [0, 255]
    rgb = np.clip(rgb, 0, 1) * 255.0

    return rgb.astype(int)


# Example RGB value
rgb = [128, 128, 128]  # Pure red
lms = rgb_to_lms(rgb)
print("LMS:", lms)

# Convert back to RGB
rgb_converted = lms_to_rgb(lms)
print("RGB:", rgb_converted)
