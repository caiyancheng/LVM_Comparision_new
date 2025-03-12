import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
torch.hub.set_dir(r'E:\Torch_hub')

# all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
#                      'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']

gaussian_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\gaussian_noise'
impulse_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\impulse_noise'
shot_noise_root_path = r'E:\Datasets\ImageNet\ImageNet-C\noise\noise\shot_noise'
clean_data_root_path = r'E:\Datasets\ImageNet\ImageNet-C\clean'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size_list = [32,32,4,32,32,4,1]
# batch_size_list = [1,32,32,4,1]
# batch_size = 4
# noise_root_path_list = [gaussian_noise_root_path, impulse_noise_root_path, shot_noise_root_path]
noise_root_path_list = [impulse_noise_root_path, shot_noise_root_path]
for noise_root_path in noise_root_path_list: #每个不同的noise模式和不同的骨架，创建一个新json和csv文件
    print(noise_root_path)
    for backbone_index in tqdm(range(len(all_backbone_list))):
        backbone_name = all_backbone_list[backbone_index]
        batch_size = batch_size_list[backbone_index]
        json_data_dict = {}

        backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
        backbone_model.eval()
        backbone_model.cuda() #输入是1,3,224,224的文件, 0-1

        noise_image_tensor_concat_list = []
        clean_image_tensor_concat_list = []
        for noise_level in tqdm(range(1,6)):
            json_data_dict[f'level_{noise_level}_cos'] = []
            level_sub_path = os.path.join(noise_root_path, str(noise_level))
            sub_dir_list = os.listdir(level_sub_path)
            for sub_dir in tqdm(sub_dir_list):
                image_filename_list = os.listdir(os.path.join(level_sub_path, sub_dir))
                for image_filename in image_filename_list:
                    noise_img_path = os.path.join(level_sub_path, sub_dir, image_filename)
                    clean_img_path = os.path.join(clean_data_root_path, sub_dir, image_filename)
                    noise_image = Image.open(noise_img_path).convert("RGB")
                    noise_image_tensor = transform(noise_image).unsqueeze(0).cuda()
                    noise_image_tensor_concat_list.append(noise_image_tensor)
                    clean_image = Image.open(clean_img_path).convert("RGB")
                    clean_image_tensor = transform(clean_image).unsqueeze(0).cuda()
                    clean_image_tensor_concat_list.append(clean_image_tensor)
                    if len(noise_image_tensor_concat_list) == batch_size:
                        noise_image_tensor_concat_tensor = torch.concat(noise_image_tensor_concat_list, 0)
                        clean_image_tensor_concat_tensor = torch.concat(clean_image_tensor_concat_list, 0)
                        noise_feature = backbone_model(noise_image_tensor_concat_tensor)
                        clean_feature = backbone_model(clean_image_tensor_concat_tensor)
                        for batch_inside_index in range(batch_size):
                            cos_similarity = float(F.cosine_similarity(noise_feature[None, batch_inside_index], clean_feature[None, batch_inside_index]).cpu())
                            json_data_dict[f'level_{noise_level}_cos'].append(cos_similarity)
                        noise_image_tensor_concat_list = []
                        clean_image_tensor_concat_list = []
                        torch.cuda.empty_cache()
            save_json_path = os.path.join(r'E:\Py_codes\LVM_Comparision\Noise_Generalization_Test\noise_test_results\dinov2', noise_root_path.split('\\')[-1], backbone_name)
            os.makedirs(save_json_path, exist_ok=True)
            with open(os.path.join(save_json_path, 'cos_sim_result.json'), 'w') as fp:
                json.dump(json_data_dict, fp)













