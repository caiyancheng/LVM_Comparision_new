import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

ILSVRC2012_root_path = r'E:\Datasets\ImageNet\ILSVRC2012_img_val'
gaussian_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\gaussian_noise'
impulse_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\impulse_noise'
shot_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\shot_noise'
new_dataset_root_path = r'E:\Datasets\ImageNet\ImageNet-C\clean'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

noise_level = 1
image_name_dir_list = []
image_name_list = []
level_sub_path = os.path.join(gaussian_noise_root_path, str(noise_level))
sub_dir_list = os.listdir(level_sub_path)
for sub_dir in tqdm(sub_dir_list):
    image_filename_list = os.listdir(os.path.join(level_sub_path, sub_dir))
    for image_filename in image_filename_list:
        image_source_path = os.path.join(ILSVRC2012_root_path, image_filename)
        image_aim_path = os.path.join(new_dataset_root_path, sub_dir, image_filename)

        img = Image.open(image_source_path).convert('RGB')
        img_transformed = transform(img)
        os.makedirs(os.path.dirname(image_aim_path), exist_ok=True)
        img_transformed.save(image_aim_path, "JPEG")
        # print(f"已保存裁剪后的图像: {image_source_path}")
