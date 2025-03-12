import abc
import os
import torch
import torch.utils.data as D
from pycvvdp.video_source_file import load_image_as_array
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import matplotlib.pyplot as plt
import numpy as np

class TID2013(D.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, path=os.path.join('quality_dataset_tests', 'tid_equijod_set'), device=None):
        super().__init__()
        self.path = path

        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.levels = torch.tensor([7, 8, 9, 10], device=self.device)
        self.images = torch.tensor([1, 2, 3, 4, 5], device=self.device)

        self.N_levels = len(self.levels)
        self.N_images = len(self.images)

        self.levels_indices, self.images_indices = torch.meshgrid(torch.arange(self.N_levels), torch.arange(self.N_images), indexing='ij')

        self.levels_indices = self.levels_indices.flatten()
        self.images_indices = self.images_indices.flatten()

        self.display_photometry = vvdp_display_photo_eotf(Y_peak=100, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([512, 384], diagonal_size_inches=30, ppd=60)

    def __getitem__(self, index):
        """
        Returns:
            vs:         video source object
            disp_photo: photometric display model
            disp_geom:  geometric display model
        """

        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'

        level = int(self.levels[self.levels_indices[index]].item())
        im = int(self.images[self.images_indices[index]].item())

        ref_path = os.path.join(self.path, 'reference.png')
        test_path = os.path.join(self.path, 'lev'+str(level)+'-im'+str(im)+'_test.png')

        reference_cont = load_image_as_array(ref_path)
        test_cont = load_image_as_array(test_path)


        vs = video_source.video_source_array( test_cont, reference_cont, 1, dim_order='HWC', display_photometry=self.display_photometry )

        return vs, self.display_photometry, self.display_geometry


    def __len__(self):
        return self.N_images * self.N_levels

    def size(self):
        return self.N_levels, self.N_images

    def plot(self, predictions, metric_name):
        error = self.error(predictions)
        fig = plt.figure(figsize=(5, 4))
        predictions = predictions.reshape(self.size())
        for prediction, level in zip(predictions, self.levels):
            plt.plot(self.images.cpu().numpy(), prediction, label='Level: '+str(int(level.cpu().item())))

        plt.xlabel('Image Distortion')
        plt.ylabel('Metric Prediction')

        title = metric_name+' [RMSE='+str(np.round(error, 3))+']'

        plt.legend()
        plt.title(title)

        fig.savefig(os.path.join('figs', 'tid2013', metric_name+'.png'))

        plt.close(fig)

    def error(self, predictions):
        predictions = predictions.reshape(self.size())
        RMSE = []
        for i in range(self.size()[1]-1):
            for j in range(i+1, self.size()[1]):
                if len(RMSE) == 0:
                    RMSE = (predictions[:, i] - predictions[:, j]) ** 2
                else:
                    RMSE += (predictions[:, i] - predictions[:, j]) ** 2
        RMSE = RMSE / ((np.max(predictions) - np.min(predictions))**2)

        return np.sqrt(np.mean(RMSE))

    def get_row_header(self):
        return ['Level', 'Distortion']

    def get_rows_conditions(self):
        return [self.levels[self.levels_indices], self.images[self.images_indices]]

    def get_preview_folder(self):
        return 'tid2013'

    def short_name(self):
        return 'TID2013 - Test'