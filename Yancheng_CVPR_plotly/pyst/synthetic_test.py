import numpy as np
from pyst.utils import *
import pyexr
import abc
from PIL import Image
from pycvvdp.video_writer import VideoWriter


class SyntheticTest:
    """
    Abstract class for all synthetic tests
    """

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    @abc.abstractmethod
    def get_test_condition_parameters(self, index):
        pass

    @abc.abstractmethod
    def get_test_condition(self, index):
        pass

    @abc.abstractmethod
    def get_condition(self, index):
        pass

    @abc.abstractmethod
    def get_row_header(self):
        pass

    @abc.abstractmethod
    def get_rows_conditions(self):
        pass

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def get_ticks(self):
        pass

    @abc.abstractmethod
    def units(self):
        pass

    @abc.abstractmethod
    def short_name(self):
        pass

    @abc.abstractmethod
    def latex_name(self):
        pass

    def preview(self, index):
        vid_source = self.get_test_condition(index)

    def save_as_exr(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        T = vid_source.get_test_frame(0, colorspace='RGB709', device=self.device)
        R = vid_source.get_reference_frame(0, colorspace='RGB709', device=self.device)

        pyexr.write(test_filename, tensor_to_numpy_image(T))
        pyexr.write(ref_filename, tensor_to_numpy_image(R))
    
    def save_as_png(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        T = Image.fromarray(tensor_to_numpy_image(vid_source.get_test_frame(0, colorspace='display_encoded_dmax', device=self.device)*255).astype(np.uint8))
        R = Image.fromarray(tensor_to_numpy_image(vid_source.get_reference_frame(0, colorspace='display_encoded_dmax', device=self.device)*255).astype(np.uint8))
            
        T.save(test_filename)
        R.save(ref_filename)

    def save_as_video(self, index, test_filename, ref_filename):
        vid_source, dm, gm = self.get_condition(index)

        _, _, N_frames = vid_source.get_video_size()
        fps = vid_source.get_frames_per_second()

        test_vw = VideoWriter(test_filename, fps=fps, codec='h265')
        ref_vw = VideoWriter(ref_filename, fps=fps, codec='h265')

        for ff in range(N_frames):
            T = tensor_to_numpy_image(vid_source.get_test_frame(ff, colorspace='display_encoded_dmax', device=self.device))
            R = tensor_to_numpy_image(vid_source.get_reference_frame(ff, colorspace='display_encoded_dmax', device=self.device))

            test_vw.write_frame_rgb(T)
            ref_vw.write_frame_rgb(R)

        test_vw.close()
        ref_vw.close()

    


        