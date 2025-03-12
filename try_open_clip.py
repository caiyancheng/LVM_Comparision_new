import open_clip
from PIL import Image
import numpy as np
models = open_clip.list_pretrained()
X = 1
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# model.eval()
#
# image_array = Image.open("CLIP_try_image.png")
# image = preprocess(image_array).unsqueeze(0)
#
# image_features = model.encode_image(image)
#
# X = 1