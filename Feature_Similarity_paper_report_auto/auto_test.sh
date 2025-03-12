#!/bin/bash

source /d/applications/anaconda/etc/profile.d/conda.sh
conda activate sam

cd /e/Py_codes/LVM_Comparision/Feature_Similarity_paper_report_auto || return

#python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_Area
#  python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Detection_Area
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Detection_Area
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_Area
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_Area
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_Area
#

#python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_Luminance
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools mae_tools --test_classes Contrast_Detection_Luminance
#python test_main.py --model_classes openclip_tools --test_classes Contrast_Detection_Luminance
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_Luminance
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_Luminance
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_Luminance
#

#python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_SpF_Gabor_Ach
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_SpF_Gabor_Ach
#

#python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_SpF_Noise_Ach
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Detection_SpF_Noise_Ach
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Detection_SpF_Noise_Ach
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_SpF_Noise_Ach
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_SpF_Noise_Ach
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_SpF_Noise_Ach
#

python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_SpF_Gabor_RG
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Detection_SpF_Gabor_RG
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Detection_SpF_Gabor_RG
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_SpF_Gabor_RG
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_SpF_Gabor_RG
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_SpF_Gabor_RG

python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Detection_SpF_Gabor_YV
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Detection_SpF_Gabor_YV
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Detection_SpF_Gabor_YV
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Detection_SpF_Gabor_YV
#python test_main.py --model_classes vae_tools --test_classes Contrast_Detection_SpF_Gabor_YV
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Detection_SpF_Gabor_YV
#

python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Masking_Phase_Coherent
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Masking_Phase_Coherent
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Masking_Phase_Coherent
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Masking_Phase_Coherent
#python test_main.py --model_classes vae_tools --test_classes Contrast_Masking_Phase_Coherent
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Masking_Phase_Coherent
#

python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Masking_Phase_Incoherent
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Masking_Phase_Incoherent
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Masking_Phase_Incoherent
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Masking_Phase_Incoherent
#python test_main.py --model_classes vae_tools --test_classes Contrast_Masking_Phase_Incoherent
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Masking_Phase_Incoherent

python test_main.py --model_classes sam_float_tools sam2_float_tools --test_classes Contrast_Matching_cos_scale_solve
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Matching_cos_scale_solve
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Matching_cos_scale_solve
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Matching_cos_scale_solve
#python test_main.py --model_classes vae_tools --test_classes Contrast_Matching_cos_scale_solve
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Matching_cos_scale_solve

#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Matching_cos_scale
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Matching_cos_scale
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Matching_cos_scale
#python test_main.py --model_classes vae_tools --test_classes Contrast_Matching_cos_scale
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Matching_cos_scale
#
#python test_main.py --model_classes no_encoder_tools dino_tools dinov2_tools --test_classes Contrast_Matching_rafal_cos
#python test_main.py --model_classes mae_tools openclip_tools --test_classes Contrast_Matching_rafal_cos
#python test_main.py --model_classes sam_tools sam2_tools --test_classes Contrast_Matching_rafal_cos
#python test_main.py --model_classes vae_tools --test_classes Contrast_Matching_rafal_cos
#python test_main.py --model_classes cvvdp_tools cvvdp_tools_rafal --test_classes Contrast_Matching_rafal_cos

echo "Script execution completed."
read -p "Press any key to continue..."