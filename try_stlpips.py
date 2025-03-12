from stlpips_pytorch import stlpips
from stlpips_pytorch import utils

path0 = r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png"
path1 = r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png"

img0 = utils.im2tensor(utils.load_image(path0))
img1 = utils.im2tensor(utils.load_image(path1))

stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")