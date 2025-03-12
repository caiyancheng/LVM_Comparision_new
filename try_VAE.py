from diffusers import AutoencoderKL

path0 = r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png"

vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")

