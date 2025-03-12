import torch
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = r"E:\Py_codes\LVM_Comparision\SAM_repo\sam2.1_hiera_large.pt"
model_cfg = r"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png")
    image_embedding = predictor.get_image_embedding()
    # masks, _, _ = predictor.predict()
    A = 1