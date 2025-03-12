import numpy as np
import os
from model_zoo import *
from test_zoo import *
from tqdm import tqdm
import torch
import gc
import argparse

def main(model_classes, test_classes):
    model_class_list = [globals()[model_class] for model_class in model_classes]
    test_class_list = [globals()[test_class] for test_class in test_classes]

    model_class_instance_list = []
    for model_class in model_class_list:
        model_instance = model_class()
        model_class_instance_list.append(model_instance)

    for test_class in tqdm(test_class_list):
        # try:
        #     test_instance = test_class(sample_num=20)
        #     test_instance.test_models(model_class_instance_list=model_class_instance_list)
        # except Exception as e:
        #     print(f"Cannot Run! Error: {e}")
        test_instance = test_class(sample_num=20)
        test_instance.test_models(model_class_instance_list=model_class_instance_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify model and test class lists.")
    parser.add_argument("--model_classes", type=str, nargs="+", help="List of model classes to use.")
    parser.add_argument("--test_classes", type=str, nargs="+", help="List of test classes to use.")
    args = parser.parse_args()

    # model_classes = ["sam2_float_tools"]
    # test_classes = ["Contrast_Detection_Luminance"]
    # model_classes = args.model_classes if args.model_classes else [
    #     "no_encoder_tools", "dino_tools", "dinov2_tools", "mae_tools", "openclip_tools", "sam_float_tools",
    #     "sam2_float_tools", "vae_tools", "cvvdp_tools_rafal"
    # ]
    model_classes = args.model_classes if args.model_classes else ["dino_tools"]
    # test_classes = args.test_classes if args.test_classes else [
    #     "Contrast_Detection_Area", "Contrast_Detection_Luminance", "Contrast_Detection_SpF_Gabor_Ach",
    #     "Contrast_Detection_SpF_Noise_Ach", "Contrast_Detection_SpF_Gabor_RG", "Contrast_Detection_SpF_Gabor_YV",
    #     "Contrast_Masking_Phase_Coherent", "Contrast_Masking_Phase_Incoherent", "Contrast_Matching_cos_scale_solve"
    # ]
    test_classes = args.test_classes if args.test_classes else ["Contrast_Matching_cos_scale_solve"]

    main(model_classes, test_classes)
