import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from SAM_repo.SAM import SAMFeatureExtractor


if __name__ == "__main__":
    feature_extractor = SAMFeatureExtractor(
        model_type="vit_h",
        checkpoint_path="SAM_repo/sam_vit_h_4b8939.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    image_path = "CLIP_try_image.png"
    features = feature_extractor.extract_features(image_path)
    print(f"Extracted features shape: {features.shape}")