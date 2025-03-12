import numpy as np
import os.path
import re
import math
import imutils
import scipy.signal as ss
#import matplotlib.pyplot as plt

#from gfxdisp.pfs import *

def decode_video_props( fname ):
    vprops = dict()
    vprops["width"]=1920
    vprops["height"]=1080

    vprops["fps"] = 24
    vprops["bit_depth"] = 8
    vprops["color_space"] = '2020'
    vprops["chroma_ss"] = '420'

    bname = os.path.splitext(os.path.basename(fname))[0]
    fp = bname.split("_")

    res_match = re.compile( '(\d+)x(\d+)p?' )

    for field in fp:

        if res_match.match( field ):
            res = field.split( "x")
            if len(res) != 2:
                raise ValueError("Cannot decode the resolution")
            vprops["width"]=int(res[0])
            vprops["height"]=int(res[1])
            continue

        if field=="444" or field=="420":
            vprops["chroma_ss"]=field

        if field=="10" or field=="10b":
            vprops["bit_depth"]=10

        if field=="8" or field=="8b":
            vprops["bit_depth"]=8

        if field=="2020" or field=="709":
            vprops["color_space"]=field

        if field=="bt709":
            vprops["color_space"]="709"

        if field=="ct2020" or field=="pq2020":
            vprops["color_space"]="2020"

    return vprops

def convert420to444(plane):

    #TODO: Replace with a proper filter
    return imutils.resize( plane, width=plane.shape[1]*2, height=plane.shape[0]*2 )

def convert444to420(comp, bit_depth):
    horfilter = np.array([[1, 6, 1]], dtype=comp.dtype)
    verfilter = np.array([[0, 1, 1]], dtype=comp.dtype).transpose()
    compF = ss.convolve2d(comp, horfilter, mode='same', boundary='symm')
    compF = compF[:,0::2]
    compF = ss.convolve2d(compF, verfilter, mode='same', boundary='symm')
    compF = compF[0::2,:]
    maxV = 2**bit_depth-1
    compF = np.right_shift(compF+8,4).clip(0,maxV)
    return compF


def fixed2float(YUV_shifted, bits):
    offset = 16/219
    weight = 1/(2**(bits-8)*219)

    YUV = np.empty(YUV_shifted.shape, dtype=np.float32)
    YUV[..., 0] = np.clip(weight*YUV_shifted[..., 0].astype(np.float32) - offset, 0, 1)

    offset = 128/224
    weight = 1/(2**(bits-8)*224)

    YUV[..., 1] = np.clip(weight*YUV_shifted[..., 1].astype(np.float32) - offset, -0.5, 0.5)
    YUV[..., 2] = np.clip(weight*YUV_shifted[..., 2].astype(np.float32) - offset, -0.5, 0.5)

    return YUV

def float2fixed(YCbCr,nbit):

    offset = (2**(nbit-8))*16
    weight = (2**(nbit-8))*219
    max_lum = (2**nbit)-1

    if nbit<=8:
        dtype = np.uint8
    else:
        dtype = np.uint16
    
    Y = np.round(weight*YCbCr[:,:,0]+offset).clip(0,max_lum).astype(dtype)
    
    offset = (2**(nbit-8)) * 128
    weight = (2**(nbit-8)) * 224  
    
    U = np.round(weight*YCbCr[:,:,1]+offset).clip(0,max_lum).astype(dtype)
    V = np.round(weight*YCbCr[:,:,2]+offset).clip(0,max_lum).astype(dtype)
  
    return np.concatenate(  (Y[:,:,np.newaxis], U[:,:,np.newaxis], V[:,:,np.newaxis]), axis=2 )




# rgb2ycbcr = np.array([[0.2126, 0.7152, 0.0722],
#                       [-0.114572, -0.385428, 0.5],
#                       [0.5, -0.454153, -0.045847]], dtype=np.float32)

ycbcr2rgb_rec709 = np.array([[1, 0, 1.5748],
                      [1, -0.18733, -0.46813],
                      [1, 1.85563, 0]], dtype=np.float32)   # This is rec 709 space
    
    
ycbcr2rgb_rec2020 = np.array([[1, 0, 1.47460],
                      [1, -0.16455, -0.57135],
                      [1, 1.88140, 0]], dtype=np.float32)   # This is rec 2020 space


class YUVFile:

    def __init__(self, file_name):        
        self.file_name = file_name

        if not os.path.isfile(file_name):
            raise FileNotFoundError( "File {} not found".format(file_name) )

        vprops = decode_video_props(file_name)
        print(vprops)

        self.bit_depth = vprops["bit_depth"]
        self.frame_bytes = int(vprops["width"]*vprops["height"])
        self.y_pixels = int(self.frame_bytes)
        self.y_shape = (vprops["height"], vprops["width"])

        if vprops["chroma_ss"]=="444":
            self.frame_bytes *= 3
            self.uv_pixels = self.y_pixels
            self.uv_shape = self.y_shape
        else: # Chroma sub-sampling
            self.frame_bytes = self.frame_bytes*3/2
            self.uv_pixels = int(self.y_pixels/4)
            self.uv_shape = (int(self.y_shape[0]/2), int(self.y_shape[1]/2))

        self.frame_pixels = self.frame_bytes
        if vprops["bit_depth"]>8:
            self.frame_bytes *= 2
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8


        self.frame_count = os.stat(file_name).st_size / self.frame_bytes
#        if math.ceil(self.frame_count)!=self.frame_count:
#            raise RuntimeError( ".yuv file does not seem to contain an integer number of frames" )

        self.frame_count = int(self.frame_count)

        self.mm = np.memmap( file_name, self.dtype, mode="r")

    def get_frame_count(self):
        return int(self.frame_count)
    
    def get_frame_yuv( self, frame_index ):

        if frame_index<0 or frame_index>=self.frame_count:
            raise RuntimeError( "The frame index is outside the range of available frames")

        offset = int(frame_index*self.frame_pixels)
        Y = self.mm[offset:offset+self.y_pixels]
        u = self.mm[offset+self.y_pixels:offset+self.y_pixels+self.uv_pixels]
        v = self.mm[offset+self.y_pixels+self.uv_pixels:offset+self.y_pixels+2*self.uv_pixels]

        return (np.reshape(Y,self.y_shape,'C'),np.reshape(u,self.uv_shape,'C'),np.reshape(v,self.uv_shape,'C'))

    # Return display-encoded (sRBG) BT.709 RGB image
    def get_frame_rgb_rec709( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)

        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb_rec709.transpose()).reshape( (*self.y_shape, 3 ), order='F' )

        return RGB

    # Return display-encoded (PQ) BT.2020 RGB image
    def get_frame_rgb_rec2020( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)

        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb_rec2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )

        return RGB

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.mm = None
 

_rgb2ycbcr_rec2020 = np.array([[0.262699236013774, 0.678001738734420, 0.059299025251807],\
  [-0.139629656645994, -0.360370861451270,  0.500000518097265],\
  [0.500000518097265,  -0.459786883720616,  -0.040213634376649]], dtype=np.float32)

_rgb2ycbcr_rec709 = np.array([[0.2126 , 0.7152 , 0.0722],\
        [-0.114572 , -0.385428 , 0.5],\
        [0.5 , -0.454153 , -0.045847]], dtype=np.float32)


class YUVWriter:

    def __init__(self, base_name, vprops):        
        self.base_name = base_name
        self.vprops = vprops
        self.fname = "{bname}_{width}x{height}_{subsampling}_{colorspace}_{bitdepth}b.yuv".format( \
            bname=base_name,\
            width=vprops["width"], height=vprops["height"], \
            subsampling=vprops["chroma_ss"], \
            colorspace=vprops["color_space"], \
            bitdepth=vprops["bit_depth"])
        
        self.pix_count = vprops["width"]*vprops["height"]
        self.bit_depth = vprops["bit_depth"]
        self.fh = open( self.fname, "w")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.fh.close()

    def append_frame_rgb( self, RGB ):

        col_mat = _rgb2ycbcr_rec2020 if self.vprops["color_space"]=='2020' else _rgb2ycbcr_rec709

        YUV = (np.reshape( RGB, (self.pix_count, 3), order='F' ) @ col_mat.transpose()).reshape( (RGB.shape), order='F' )

        YUV_fixed = float2fixed( YUV, self.bit_depth )

        Y = YUV_fixed[:,:,0]
        if self.vprops["chroma_ss"] == '444':
            u = YUV_fixed[:,:,1]
            v = YUV_fixed[:,:,2]
        elif self.vprops["chroma_ss"] == '420':
            u = convert444to420(YUV_fixed[:,:,1], self.bit_depth)
            v = convert444to420(YUV_fixed[:,:,2], self.bit_depth)
        else:
            raise RuntimeError( 'Not implemented' )

        Y.tofile(self.fh)
        u.tofile(self.fh)
        v.tofile(self.fh)

    '''
    Depreciated. Use `append_frame_rgb` instead. 
    '''
    def append_frame_rgb_rec2020( self, RGB ):

        YUV = (np.reshape( RGB, (self.pix_count, 3), order='F' ) @ _rgb2ycbcr_rec2020.transpose()).reshape( (RGB.shape), order='F' )

        YUV_fixed = float2fixed( YUV, self.bit_depth )

        Y = YUV_fixed[:,:,0]
        u = convert444to420(YUV_fixed[:,:,1], self.bit_depth)
        v = convert444to420(YUV_fixed[:,:,2], self.bit_depth)

        Y.tofile(self.fh)
        u.tofile(self.fh)
        v.tofile(self.fh)