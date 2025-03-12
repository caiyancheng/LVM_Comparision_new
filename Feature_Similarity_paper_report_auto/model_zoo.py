import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
torch.hub.set_dir(r'E:\Torch_hub')
from display_encoding import display_encode
display_encode_tool = display_encode(400)
from transformers import AutoImageProcessor, ViTMAEForPreTraining
import open_clip
from SAM_repo.SAM import SAMFeatureExtractor
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers import AutoencoderKL
import pycvvdp
import lpips
from stlpips_pytorch import stlpips
import torch.nn.functional as F
from thop import profile
import torchprofile
#要求输入的image_array是linear luminance space, (224,224,3)
import torch.nn as nn

def check_input(input, aim_shape=(224,224,3)):
    if input.shape == aim_shape:
        return 1
    else:
        raise ValueError('Input does not have the shape of (224, 224, 3)')

class no_encoder_tools:
    def __init__(self):
        self.name = 'no_encoder'
        self.backbone_list = [self.name]

    def load_pretrained(self, backbone_name):
        X = 1

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = image_C_tensor
        return feature

class cvvdp_tools:
    def __init__(self):
        self.name = 'cvvdp'
        self.cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
        self.backbone_list = [self.name]

    def load_pretrained(self, backbone_name):
        X = 1

    def compute_score(self, T_L_array, R_L_array): #直接输入Luminance
        check_input(T_L_array)
        check_input(R_L_array)
        T_L_tensor = torch.tensor((display_encode_tool.L2C_sRGB(T_L_array) * 255).astype(np.uint8))
        R_L_tensor = torch.tensor((display_encode_tool.L2C_sRGB(R_L_array) * 255).astype(np.uint8))
        JOD, m_stats = self.cvvdp.predict(T_L_tensor, R_L_tensor, dim_order="HWC")
        return JOD

class cvvdp_tools_rafal:
    def __init__(self):
        self.name = 'cvvdp_hdr'
        disp_photo = pycvvdp.vvdp_display_photo_eotf(400, contrast=1000000, source_colorspace='BT.709', EOTF="linear", E_ambient=0)
        self.metric = pycvvdp.cvvdp(display_name='standard_hdr_linear', display_photometry=disp_photo)
        self.backbone_list = [self.name]

    def load_pretrained(self, backbone_name):
        X = 1

    def compute_score(self, T_L_array, R_L_array): #直接输入Luminance
        check_input(T_L_array)
        check_input(R_L_array)
        T_L_tensor = torch.tensor(T_L_array, dtype=torch.float32)
        R_L_tensor = torch.tensor(R_L_array, dtype=torch.float32)
        JOD, m_stats = self.metric.predict(T_L_tensor, R_L_tensor, dim_order="HWC")
        return JOD

class dino_tools:
    def __init__(self):
        self.name = 'dino'
        self.backbone_list = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16',
                     'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']
    def load_pretrained(self, backbone_name):
        backbone_model = torch.hub.load('facebookresearch/dino:main', backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        return backbone_model

    def compute_params_flops(self):
        for dino_index in self.backbone_list:
            model = torch.hub.load('facebookresearch/dino:main', dino_index)
            model.eval()
            model.cuda()
            input_tensor = torch.randn(224,224,3).cuda()
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            params_num = sum(p.numel() for p in model.parameters()) / 1e6
            flops, params = profile(model, inputs=(input_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'dino_model_name: {dino_index},'
                  f' params_all: {params_num:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = backbone_model(image_C_tensor)
        return feature

class dinov2_tools:
    def __init__(self):
        self.name = 'dinov2'
        self.backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                         'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
    def load_pretrained(self, backbone_name):
        backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        return backbone_model

    def compute_params_flops(self):
        for dinov2_index in self.backbone_list:
            model = torch.hub.load('facebookresearch/dinov2', dinov2_index)
            model.eval()
            model.cuda()
            input_tensor = torch.randn(224,224,3).cuda()
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            params_num = sum(p.numel() for p in model.parameters()) / 1e6
            flops, params = profile(model, inputs=(input_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'dino_model_name: {dinov2_index},'
                  f' params_all: {params_num:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        feature = backbone_model(image_C_tensor)
        return feature

class mae_tools:
    def __init__(self):
        self.name = 'mae'
        self.backbone_list = ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge']
        torch.manual_seed(8)
        self.default_noise = torch.rand(1, 196)
    def load_pretrained(self, backbone_name):
        processor = AutoImageProcessor.from_pretrained(f'facebook/{backbone_name}')
        model = ViTMAEForPreTraining.from_pretrained(f'facebook/{backbone_name}')
        model.eval()
        return (processor, model)

    def compute_params_flops(self):
        for mae_index in self.backbone_list:
            processor = AutoImageProcessor.from_pretrained(f'facebook/{mae_index}')
            model = ViTMAEForPreTraining.from_pretrained(f'facebook/{mae_index}')
            model.eval()
            input_tensor = np.array(torch.randn(224,224,3))
            input_dict = processor(images=input_tensor.astype(np.float32), return_tensors="pt", do_resize=False,
                                   do_rescale=False,
                                   do_normalize=False)
            input_dict['noise'] = self.default_noise
            params_num = sum(p.numel() for p in model.vit.parameters()) / 1e6
            flops, params = profile(model.vit, inputs=(input_dict['pixel_values'],input_dict['noise'],))
            flops = flops / 1e9
            params = params / 1e6
            print(f'mae_model_name: {mae_index},'
                  f' params_all: {params_num:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        processor, model = backbone_model
        input_dict = processor(images=image_C_array.astype(np.float32), return_tensors="pt", do_resize=False, do_rescale=False,
                               do_normalize=False)
        input_dict['noise'] = self.default_noise
        feature = model.vit(**input_dict).last_hidden_state
        return feature

class openclip_tools:
    def __init__(self):
        self.name = 'openclip'
        clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'),
                       ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'),
                       ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
                       ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]
        self.backbone_list = clip_model_list

    def load_pretrained(self, backbone_name):
        clip_model_name, clip_model_trainset = backbone_name
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                     pretrained=clip_model_trainset,
                                                                     cache_dir=r'E:\Openclip_cache')
        model.eval()
        model.cuda()
        return (model, preprocess)

    def compute_params(self):
        for openclip_index in self.backbone_list:
            clip_model_name, clip_model_trainset = openclip_index
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                         pretrained=clip_model_trainset,
                                                                         cache_dir=r'E:\Openclip_cache')
            params_num = sum(p.numel() for p in model.parameters())
            print(f'clip_model_name: {clip_model_name}, clip_model_trainset: {clip_model_trainset}, params_num: {params_num}')

    def compute_params_flops(self):
        for openclip_index in self.backbone_list:
            clip_model_name, clip_model_trainset = openclip_index
            model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name,
                                                                         pretrained=clip_model_trainset,
                                                                         cache_dir=r'E:\Openclip_cache')
            model.eval()
            model.cuda()
            image_encoder = model.visual
            input_tensor = torch.randn(224, 224, 3).cuda()
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            params_num = sum(p.numel() for p in image_encoder.parameters()) / 1e6
            flops, params = profile(image_encoder, inputs=(input_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'clip_model_name: {clip_model_name}, clip_model_trainset: {clip_model_trainset},'
                  f' params_all: {params_num:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        model, preprocess = backbone_model
        feature = model.encode_image(image_C_tensor)
        return feature

class sam_tools:
    def __init__(self):
        self.name = 'sam'
        self.backbone_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
        self.sam_vit_list = ['vit_b', 'vit_l', 'vit_h']
    def load_pretrained(self, backbone_name):
        model_type = backbone_name.split('_')[1]+'_'+backbone_name.split('_')[2]
        backbone_model = SAMFeatureExtractor(
            model_type=model_type,
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{backbone_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return backbone_model

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = (display_encode_tool.L2C_sRGB(image_L_array) * 255).astype(np.uint8)
        feature = backbone_model.extract_features_from_numpy(image_C_array)
        return feature

class sam_float_tools:
    def __init__(self):
        self.name = 'sam_float'
        self.backbone_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
        self.sam_vit_list = ['vit_b', 'vit_l', 'vit_h']
    def load_pretrained(self, backbone_name):
        model_type = backbone_name.split('_')[1]+'_'+backbone_name.split('_')[2]
        backbone_model = SAMFeatureExtractor(
            model_type=model_type,
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{backbone_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return backbone_model

    def compute_params_flops(self):
        for backbone_name in self.backbone_list:
            model_type = backbone_name.split('_')[1]+'_'+backbone_name.split('_')[2]
            backbone_model = SAMFeatureExtractor(
                model_type=model_type,
                checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{backbone_name}.pth",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            input_array = np.array(torch.randn(224,224,3)).astype(np.float32)
            image_tensor = torch.from_numpy(input_array).permute(2, 0, 1)[None, ...].cuda()
            image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_encoder = backbone_model.predictor.model.image_encoder
            params_num = sum(p.numel() for p in image_encoder.parameters()) / 1e6
            flops, params = profile(image_encoder, inputs=(image_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'sam_model_name: {backbone_name},'
                  f' params_all: {params_num:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array).astype(np.float32)
        feature = backbone_model.extract_features_from_numpy_float32(image_C_array)
        return feature

class sam2_tools:
    def __init__(self):
        self.name = 'sam2'
        self.backbone_list = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
        self.sam_config_list = ['sam2.1_hiera_t', 'sam2.1_hiera_s', 'sam2.1_hiera_b+', 'sam2.1_hiera_l']
    def load_pretrained(self, backbone_name):
        backbone_index = self.backbone_list.index(backbone_name)
        sam_config_name = self.sam_config_list[backbone_index]
        checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{backbone_name}.pt"
        model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
        backbone_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        return backbone_model
    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = (display_encode_tool.L2C_sRGB(image_L_array) * 255).astype(np.uint8)
        backbone_model.set_image(image_C_array)
        feature = backbone_model.get_image_embedding()
        return feature

class sam2_float_tools:
    def __init__(self):
        self.name = 'sam2_float'
        self.backbone_list = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
        self.sam_config_list = ['sam2.1_hiera_t', 'sam2.1_hiera_s', 'sam2.1_hiera_b+', 'sam2.1_hiera_l']
    def load_pretrained(self, backbone_name):
        backbone_index = self.backbone_list.index(backbone_name)
        sam_config_name = self.sam_config_list[backbone_index]
        checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{backbone_name}.pt"
        model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
        backbone_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        return backbone_model

    def compute_params_flops(self):
        for backbone_name, sam_config_name in zip(self.backbone_list, self.sam_config_list):
            checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{backbone_name}.pt"
            model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
            backbone_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

            input_array = np.array(torch.randn(224,224,3)).astype(np.float32)
            image_tensor = torch.from_numpy(input_array).permute(2, 0, 1)[None, ...].cuda()
            image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)

            model_1 = backbone_model.model
            image_encoder = model_1.image_encoder
            params_num_1 = sum(p.numel() for p in model_1.parameters()) / 1e6
            params_num_encoder = sum(p.numel() for p in image_encoder.parameters()) / 1e6

            # backbone_model.set_image_yancheng_float(image_tensor)
            # feature = backbone_model.get_image_embedding()
            flops, params = profile(image_encoder, inputs=(image_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'sam_model_name: {backbone_name},'
                  f' params_all: {params_num_1:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')

    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array).astype(np.float32)
        image_tensor = torch.from_numpy(image_C_array).permute(2, 0, 1)[None,...].cuda()
        image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
        backbone_model.set_image_yancheng_float(image_tensor)
        feature = backbone_model.get_image_embedding()
        return feature

class vae_tools:
    def __init__(self):
        self.name = 'vae'
        self.backbone_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0']
    def load_pretrained(self, backbone_name):
        backbone_model = AutoencoderKL.from_pretrained(backbone_name, subfolder="vae")
        return backbone_model

    def compute_params_flops(self):
        for backbone_name in self.backbone_list:
            backbone_model = AutoencoderKL.from_pretrained(backbone_name, subfolder="vae").cuda()
            input_tensor = torch.randn(224, 224, 3)
            image_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            latent = backbone_model.encode(image_tensor)
            feature = latent.latent_dist.sample()

            image_encoder = backbone_model.encoder
            params_num_1 = sum(p.numel() for p in image_encoder.parameters()) / 1e6
            flops, params = profile(image_encoder, inputs=(image_tensor,))
            flops = flops / 1e9
            params = params / 1e6
            print(f'vae_model_name: {backbone_name},'
                  f' params_all: {params_num_1:.2f} MB, params_forward: {params:.2f} MB, GFlops: {flops:.2f}')


    def forward_feature(self, backbone_model, image_L_array):
        check_input(image_L_array)
        image_C_array = display_encode_tool.L2C_sRGB(image_L_array)
        image_C_tensor = torch.tensor(image_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        norm_image_C_tensor = (image_C_tensor - 0.5) * 2
        latent = backbone_model.encode(norm_image_C_tensor)
        feature = latent.latent_dist.sample()
        return feature




# class lpips_tools:
#     def __init__(self):
#         self.name = 'lpips'
#         self.backbone_list = ['alex', 'vgg']
#         self.loss_fn_list = []
#         for lpips_model in self.backbone_list:
#             loss_fn = lpips.LPIPS(net=lpips_model).eval()
#             self.loss_fn_list.append(loss_fn)
#     def compute_score(self, T_L_array, R_L_array):
#         check_input(T_L_array)
#         check_input(R_L_array)
#         T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
#         R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
#         T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...]
#         R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...]
#         norm_T_C_tensor = (T_C_tensor - 0.5) * 2
#         norm_R_C_tensor = (R_C_tensor - 0.5) * 2
#         loss_list = []
#         for loss_fn in self.loss_fn_list:
#             loss_value = float(loss_fn(norm_T_C_tensor, norm_R_C_tensor))
#             loss_list.append(loss_value)
#         return loss_list
#
# class stlpips_tools:
#     def __init__(self):
#         self.name = 'stlpips'
#         self.backbone_list = ['alex', 'vgg']
#         self.loss_fn_list = []
#         for stlpips_model in self.backbone_list:
#             loss_fn = stlpips.LPIPS(net=stlpips_model, variant="shift_tolerant").eval()
#             self.loss_fn_list.append(loss_fn)
#     def compute_score(self, T_L_array, R_L_array):
#         check_input(T_L_array)
#         check_input(R_L_array)
#         T_C_array = display_encode_tool.L2C_sRGB(T_L_array)
#         R_C_array = display_encode_tool.L2C_sRGB(R_L_array)
#         T_C_tensor = torch.tensor(T_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...]
#         R_C_tensor = torch.tensor(R_C_array, dtype=torch.float32).permute(2, 0, 1)[None, ...]
#         norm_T_C_tensor = (T_C_tensor - 0.5) * 2
#         norm_R_C_tensor = (R_C_tensor - 0.5) * 2
#         loss_list = []
#         for loss_fn in self.loss_fn_list:
#             loss_value = float(loss_fn(norm_T_C_tensor, norm_R_C_tensor))
#             loss_list.append(loss_value)
#         return self.backbone_list, loss_list