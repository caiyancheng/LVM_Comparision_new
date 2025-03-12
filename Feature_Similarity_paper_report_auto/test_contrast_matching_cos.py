import os
import numpy as np
from model_zoo import *
# model_dict = {'dino': ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16',
#                        'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50'],
#               'dinov2': ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov2_vits14_reg',
#                          'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'],
#               'mae': ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge'],
#               'openclip': [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
#                        ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'),
#                        ('ViT-B-16', 'laion2b_s34b_b88k'),
#                        ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'),
#                        ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
#                        ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
#                        ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')],
#               'sam': ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939'],
#               'sam2': ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large'],
#               }
model_class_list = [no_encoder_tools, dino_tools, dinov2_tools, mae_tools, openclip_tools, sam_tools, sam2_tools,
                    vae_tools, cvvdp_tools, lpips_tools, stlpips_tools]