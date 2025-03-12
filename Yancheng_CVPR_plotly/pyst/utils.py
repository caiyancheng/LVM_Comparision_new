import torch 
from math import pi, log
from pycvvdp.utils import PU

def generate_gabor_patch(width, height, frames, ppd, fps, contrast, s_freq, t_freq, orientation, luminance, dkl_col_direction, ge_sigma, eccentricity, phase=0, device=None):

    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) - eccentricity , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    
    rr = xx * torch.cos(torch.deg2rad(-orientation)) + yy * torch.sin(torch.deg2rad(-orientation))

    grating = torch.sin(2*pi*rr*s_freq + torch.deg2rad(torch.tensor(phase))) * contrast * luminance
    sine_grating = grating.view(height, width, 1, 1)

    t = torch.linspace(0, frames/fps, frames).to(device)
    temporal_sine = torch.cos(2*pi*t*t_freq).view(1, 1, 1, frames) # This is giving us H,W,C,F maybe we will want to change that

    R2 = xx**2 + yy**2
    ge =  torch.exp(-R2/(2*ge_sigma**2)).view(height, width, 1, 1)  # Create the gaussian envelope

    test_stimuli = sine_grating * ge * temporal_sine

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)

    DKL_reference = torch.ones(height, width, 3, frames).to(device) * DKL_gray
    DKL_test = DKL_reference + test_stimuli * dkl_col_direction.view(1, 1, 3, 1)

    XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
    XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))

    return XYZ_test, XYZ_reference

def generate_gabor_mask(width, height, ppd, test_contrast, masker_contrast, s_freq, test_orientation, masker_orientation, luminance, test_dkl_col_direction, masker_dkl_col_direction, ge_sigma, device=None):
    
    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    # We are using a cos grating! 

    # Create the masker!

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(0, width_deg, width).to(device) , torch.linspace(0, height_deg, height).to(device), indexing='xy')
    
    dd = torch.sqrt(xx**2 + yy**2)
    cosB = ( xx * torch.sin(torch.deg2rad(-masker_orientation)) + yy * torch.cos(torch.deg2rad(-masker_orientation)) ) / dd
    cosB[0, 0] = 0
    d = torch.sqrt(1 - cosB**2) * dd 

    grating = torch.cos(2*pi*d*s_freq) * masker_contrast * luminance
    masker_grating = grating.view(height, width, 1, 1) 

    # Create the target!

    xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    
    rr = xx * torch.cos(torch.deg2rad(-test_orientation)) + yy * torch.sin(torch.deg2rad(-test_orientation))

    #grating = torch.cos(2*pi*rr*s_freq) * test_contrast * luminance
    grating = torch.cos(2*pi*d*s_freq) * test_contrast * luminance
    sine_grating = grating.view(height, width, 1, 1)

    R2 = xx**2 + yy**2
    ge =  torch.exp(-R2/(2*ge_sigma**2)).view(height, width, 1, 1)  # Create the gaussian envelope

    target_grating = sine_grating * ge

    # Get the stimuli!

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)
    DKL_reference = torch.ones(height, width, 3, 1).to(device) * DKL_gray


    DKL_target = target_grating * test_dkl_col_direction.view(1, 1, 3, 1)
    DKL_masker = DKL_reference + masker_grating * masker_dkl_col_direction.view(1, 1, 3, 1)

    XYZ_masker = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_masker.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
    XYZ_target = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_target.permute(2, 3, 0, 1)))

    return XYZ_target, XYZ_masker


def create_cycdeg_image(height, width, ppd, device=None):

    KX = (torch.remainder(0.5 + torch.arange(width).to(device) / width, 1) - 0.5) * ppd
    KY = (torch.remainder(0.5 + torch.arange(height).to(device) / height, 1) - 0.5) * ppd

    XX, YY = torch.meshgrid(KX, KY, indexing='xy')

    D = torch.sqrt(torch.add(torch.pow(XX, 2), torch.pow(YY, 2)))

    return D


def generate_band_limited_noise(width, height, ppd, contrast, s_freq, band_width, luminance, dkl_col_direction, frames, device=None):

    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    # Fix Seed
    torch.manual_seed(2)

    noise = torch.randn((height, width)).to(device)
    noise_f = torch.fft.fft2(noise)

    rho = create_cycdeg_image(height, width, ppd, device=device)

    freq_edge_low = torch.pow(2, torch.log2(s_freq) - 0.5)
    freq_edge_high = torch.pow(2, torch.log2(s_freq) + 0.5)

    noise_f[(rho<freq_edge_low) | (rho>freq_edge_high)] = 0

    test_stimuli = torch.real(torch.fft.ifft2(noise_f)).view(height, width, 1, 1)
    test_stimuli /= torch.std(test_stimuli)

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)

    DKL_reference = torch.ones(height, width, 3, frames).to(device) * DKL_gray
    DKL_test = DKL_reference + test_stimuli * luminance * contrast * dkl_col_direction.view(1, 1, 3, 1) 

    XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
    XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))

    return XYZ_test, XYZ_reference


def generate_flat_noise_mask(width, height, ppd, test_contrast, masker_contrast, s_freq, test_orientation, luminance, test_dkl_col_direction, masker_dkl_col_direction, ge_sigma, device=None):
    
    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    # Fix Seed
    torch.manual_seed(2)

    # Create the masker! 

    noise = torch.randn((height, width)).to(device)
    masker_noise_f = torch.fft.fft2(noise)

    rho = create_cycdeg_image(height, width, ppd, device=device)
    masker_noise_f[rho > 12] = 0

    masker_stimuli = torch.real(torch.fft.ifft2(masker_noise_f)).view(height, width, 1, 1)
    masker_stimuli /= torch.std(masker_stimuli)
    masker_stimuli = masker_stimuli * luminance * masker_contrast

    # Create the target! (we are using a cosine grating!)

    width_deg = width / ppd
    height_deg = height / ppd

    #xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    xx, yy = torch.meshgrid(torch.linspace(0, width_deg, width).to(device) , torch.linspace(0, height_deg, height).to(device), indexing='xy')

    rr = xx * torch.cos(torch.deg2rad(-test_orientation)) + yy * torch.sin(torch.deg2rad(-test_orientation))
    grating = torch.cos(2*pi*rr*s_freq) * test_contrast * luminance
    sine_grating = grating.view(height, width, 1, 1)

    xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    R2 = xx**2 + yy**2
    ge =  torch.exp(-R2/(2*ge_sigma**2)).view(height, width, 1, 1)  # Create the gaussian envelope

    target_stimuli = sine_grating * ge

    # Get the stimuli!

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)
    DKL_reference = torch.ones(height, width, 3, 1).to(device) * DKL_gray


    DKL_target = target_stimuli * test_dkl_col_direction.view(1, 1, 3, 1)
    DKL_masker = DKL_reference + masker_stimuli * masker_dkl_col_direction.view(1, 1, 3, 1)

    XYZ_masker = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_masker.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
    XYZ_target = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_target.permute(2, 3, 0, 1)))

    return XYZ_target, XYZ_masker


def generate_sinusoidal_grating(width, height, frames, ppd, fps, contrast, s_freq, luminance, dkl_col_direction, phase=0, device=None):

    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(-width_deg/2, width_deg/2, width).to(device) , torch.linspace(-height_deg/2, height_deg/2, height).to(device), indexing='xy')
    
    rr = xx

    grating = torch.sin(2*pi*rr*s_freq + torch.deg2rad(torch.tensor(phase))) * contrast * luminance
    sine_grating = grating.view(height, width, 1, 1)

    test_stimuli = sine_grating

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)

    DKL_reference = torch.ones(height, width, 3, frames).to(device) * DKL_gray
    DKL_test = DKL_reference + test_stimuli * dkl_col_direction.view(1, 1, 3, 1)

    XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1))) # FROM HWCF to CFHW
    XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))

    return XYZ_test, XYZ_reference


def generate_sine_wave(width, height, frames, ppd, fps, contrast, s_freq, luminance, dkl_col_direction, phase=0, device=None):
    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    sine_wave = torch.sin(torch.linspace(0, 2*pi*s_freq*height/ppd, height)).to(device)
    sine_wave = (sine_wave>0).to(sine_wave.dtype)*2-1
    test_stimuli = sine_wave.view(height, 1, 1, 1) * luminance * contrast

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)

    DKL_reference = torch.ones(height, width, 3, frames).to(device) * DKL_gray
    DKL_test = DKL_reference + test_stimuli * dkl_col_direction.view(1, 1, 3, 1)

    XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1)))  # FROM HWCF to CFHW
    XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))

    return XYZ_test, XYZ_reference


def generate_disk(width, height, frames, ppd, fps, contrast, s_freq, t_freq, orientation, luminance,
                         dkl_col_direction, ge_sigma, eccentricity, phase=0, device=None):

    wp_d65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

    width_deg = width / ppd
    height_deg = height / ppd

    xx, yy = torch.meshgrid(torch.linspace(-width_deg / 2, width_deg / 2, width).to(device) - eccentricity,
                            torch.linspace(-height_deg / 2, height_deg / 2, height).to(device), indexing='xy')

    disk = xx**2 + yy**2 - ge_sigma**2
    disk = (disk <=0 ).to(disk.dtype) * luminance * contrast
    disk = disk.view(height, width, 1, 1)

    t = torch.linspace(0, frames / fps, frames).to(device)
    temporal_sine = torch.sin(2 * pi * t * t_freq).view(1, 1, 1, frames)  # This is giving us H,W,C,F maybe we will want to change that

    test_stimuli = disk * temporal_sine

    DKL_gray = LMS2DKL_D65(XYZ2LMS2006(wp_d65 * luminance)).view(1, 1, 3, 1)

    DKL_reference = torch.ones(height, width, 3, frames).to(device) * DKL_gray
    DKL_test = DKL_reference + test_stimuli * dkl_col_direction.view(1, 1, 3, 1)

    XYZ_reference = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_reference.permute(2, 3, 0, 1)))  # FROM HWCF to CFHW
    del DKL_reference
    XYZ_test = LMS20062XYZ_nd(DKL2LMS_D65_nd(DKL_test.permute(2, 3, 0, 1)))
    del DKL_test

    return XYZ_test, XYZ_reference

def Yxy2XYZ(Yxy):

    # We will assume that Yxy is a tensor of the sahpe X,C

    X = Yxy[:,0] * Yxy[:,1] / Yxy[:,2]
    Y = Yxy[:, 0]
    Z = Yxy[:,0] / Yxy[:,2] * (1-Yxy[:,1]-Yxy[:,2])

    return torch.stack((X, Y, Z), dim=-1)

def XYZ2LMS2006(XYZ):

    # We expect a vector of shape X,C 

    M_xyz_lms2006 = torch.tensor([ 
        [0.187596268556126, 0.585168649077728, -0.026384263306304],
        [-0.133397430663221, 0.405505777260049, 0.034502127690364],
        [0.000244379021663, -0.000542995890619, 0.019406849066323 ]], device=XYZ.device)
    
    return torch.matmul(XYZ, M_xyz_lms2006.T)


def LMS2DKL_D65(LMS):

    # We expect a vector of shape X,C

    lms_gray = [0.739876529525622,   0.320136241543338,   0.020793708751515]

    mc1 = lms_gray[0]/lms_gray[1]
    mc2 = (lms_gray[0]+lms_gray[1])/lms_gray[2]

    M_lms2006_dkl = torch.tensor([
        [1, 1, 0],
        [1, -mc1, 0],
        [-1, -1, mc2]], device=LMS.device)

    return torch.matmul(LMS, M_lms2006_dkl.T)


def DKL2LMS_D65_nd(DKL):

    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    lms_gray = [0.739876529525622,   0.320136241543338,   0.020793708751515]

    mc1 = lms_gray[0]/lms_gray[1]
    mc2 = (lms_gray[0]+lms_gray[1])/lms_gray[2]

    M_lms2006_dkl = torch.tensor([
        [1, 1, 0],
        [1, -mc1, 0],
        [-1, -1, mc2]], device=DKL.device)

    M_dkl_lms2006 = M_lms2006_dkl.inverse()

    return torch.matmul(M_dkl_lms2006, DKL.reshape(3, -1)).reshape(DKL.size())


def LMS20062XYZ_nd(LMS):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_lms2006_xyz = torch.tensor([
        [2.629129278399650, -3.780202391780134, 10.294956387893450],
        [0.865649062438827, 1.215555811642301, -0.984175688105352],
        [-0.008886561474676, 0.081612628990755, 51.371024830897888 ]], device=LMS.device)
    
    return torch.matmul(M_lms2006_xyz, LMS.reshape(3, -1)).reshape(LMS.size())

def XYZ2RGB709_nd(XYZ):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_xyz2rgb = torch.tensor([ 
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]], device=XYZ.device)

    return torch.matmul(M_xyz2rgb, XYZ.reshape(3, -1)).reshape(XYZ.size())

def XYZ2RGB2020_nd(XYZ):
    # We expect the input to be a multi-dimensional tensor of the shape (C,...)

    M_xyz2rgb = torch.tensor([
        [0.636953507, 0.144619185, 0.168855854],
        [0.262698339, 0.678008766, 0.0592928953],
        [4.99407097e-17, 0.0280731358, 1.06082723]], device=XYZ.device).inverse()

    return torch.matmul(M_xyz2rgb, XYZ.reshape(3, -1)).reshape(XYZ.size())

def lin2srgb(L):

    t = 0.0031308
    a = 0.055
    L = L.clip(min=0, max=1)
    p = torch.where(L <= t, L * 12.92, (1 + a) * L ** (1 / 2.4) - a)
    return p

def tensor_to_numpy_image(T):
    return torch.squeeze(T.permute((3,4,1,0,2)), dim=(3,4)).cpu().numpy()

def lin2pq( L ):
    """ Convert from absolute linear values (between 0.005 and 10000) to PQ-encoded values V (between 0 and 1)
    """
    Lmax = 10000
    #Lmin = 0.005
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000
    im_t = (torch.clip(L,0,Lmax)/Lmax) ** n
    V  = ((c2*im_t + c1) / (1+c3*im_t)) ** m
    return V

def lin2hlg(L, Lmax=10000):

    L = (L/Lmax).clip(min=0, max=1)

    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * log(4 * a)

    # OETF
    V = torch.where(
        L <= 1/12,
        torch.sqrt(3*L),
        a * torch.log(12*L - b) + c
    )

    return V


def display_encode(L, display_photometry):

    EOTF = display_photometry.EOTF
    Lmax = display_photometry.get_peak_luminance()

    if EOTF == 'sRGB':
        luma = lin2srgb(L/Lmax)
    elif EOTF == 'PQ':
        luma = lin2pq(L)
    elif EOTF == 'HLG':
        luma = lin2hlg(L, Lmax)
    elif EOTF == 'linear':
        PU_obj = PU()
        PU_max = PU_obj.encode(torch.as_tensor(Lmax))
        luma = PU_obj.encode(L) / PU_max 
    
    return luma


def RGB2Y(I, display_photometry):

    # The shape of the input is BCHW

    rgb2y_matrix = torch.tensor( display_photometry.rgb2xyz_list[1], dtype=I.dtype, device=I.device )
    
    Y = torch.sum(I*(rgb2y_matrix.view(1,3,1,1)), dim=-3, keepdim=True)

    return Y


# def display_encode(L, display_photometry):

#     EOTF = display_photometry.EOTF
#     Lmax = display_photometry.get_peak_luminance()

#     if EOTF == 'sRGB':
#         luma = lin2srgb(L/Lmax)
#     elif EOTF == 'PQ':
#         luma = lin2pq(L)
#     elif EOTF == 'HLG':
#         luma = lin2hlg(L, Lmax)
#     elif EOTF == 'linear':
#         PU_obj = PU()
#         PU_max = PU_obj.encode(torch.as_tensor(Lmax))
#         luma = PU_obj.encode(L) / PU_max 
    
#     return luma

