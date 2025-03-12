import pycvvdp

path0 = r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png"
path1 = r"E:\Py_codes\LVM_Comparision/CLIP_try_image.png"

I_ref = pycvvdp.load_image_as_array(path0)
I_test = pycvvdp.load_image_as_array(path1)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')
JOD, m_stats = cvvdp.predict( I_test, I_ref, dim_order="HWC" )

x = 1