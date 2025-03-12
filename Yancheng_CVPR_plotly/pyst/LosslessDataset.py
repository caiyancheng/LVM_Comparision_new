import abc
import os
import torch
import torch.utils.data as D
from pycvvdp.video_source_file import load_image_as_array
from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr
import pandas as pd
from scipy.optimize import minimize_scalar

class LosslessDataset(D.Dataset, metaclass=abc.ABCMeta):

    def __init__(self, path=os.path.join('quality_dataset_tests', 'vis_lossess_view_cond', 'images'), device=None):
        super().__init__()
        self.path = path

        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # gt_path = os.path.join('quality_dataset_tests', 'x_distortion_levels.json')
        # with open(gt_path, 'r') as fp:
        #     self.data = json.load(fp)

        # self.probabilities = torch.tensor([0.25, 0.5, 0.75], device=self.device)

        # gt_path = os.path.join('quality_dataset_tests', 'vis_lossess_view_cond', 'data_averaged', 'vlt_clean.csv')
        # self.data = pd.read_csv(gt_path)

        # self.probabilities = torch.tensor([0.25], device=self.device)

        gt_path = os.path.join('quality_dataset_tests', 'vlts_0.25.json')
        with open(gt_path, 'r') as fp:
            self.data = json.load(fp)['vlt_30']

        self.distortions = torch.linspace(2, 98, 49, device=self.device)
        self.images = torch.linspace(0, 19, 20, device=self.device)

        self.N_dis = len(self.distortions)
        self.N_images = len(self.images)

        self.dis_indices, self.images_indices = torch.meshgrid(torch.arange(self.N_dis), torch.arange(self.N_images), indexing='ij')

        self.dis_indices = self.dis_indices.flatten()
        self.images_indices = self.images_indices.flatten()

        self.display_photometry = vvdp_display_photo_eotf(Y_peak=220, contrast=1000, source_colorspace='sRGB')
        self.display_geometry = vvdp_display_geometry([512, 512], diagonal_size_inches=30, ppd=30)

    def __getitem__(self, index):
        """
        Returns:
            vs:         video source object
            disp_photo: photometric display model
            disp_geom:  geometric display model
        """

        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'


        distortion_level = int(self.distortions[self.dis_indices[index]].item())
        im = int(self.images[self.images_indices[index]].item())

        # if prob == 0.25:
        #     index = 0
        # elif prob == 0.5:
        #     index = 1
        # elif prob == 0.75:
        #     index = 2

        if im < 10:
            im_name = 'i'+str(im)+'webp'
        else:
            im_name = 'i'+str(im)+'jpeg'

        #distortion_level = int(self.data[im_name]['Lum_220_PPD_60'][index])

        # all_possible_dis = np.linspace(0, 100, 51)
        # distortion_level = self.data['q_220_60'][im]
        # distortion_level = int(all_possible_dis[np.argmin(np.abs(distortion_level - all_possible_dis))])

        ref_path = os.path.join(self.path, im_name+'_ref.png')
        test_path = os.path.join(self.path, im_name+'_'+str(distortion_level)+'.png')

        reference_cont = load_image_as_array(ref_path)
        test_cont = load_image_as_array(test_path)

        if test_cont.shape[2] == 1:
            test_cont = np.repeat(test_cont, 3, 2)

        vs = video_source.video_source_array( test_cont, reference_cont, 1, dim_order='HWC', display_photometry=self.display_photometry )

        return vs, self.display_photometry, self.display_geometry


    def __len__(self):
        return self.N_images * self.N_dis

    def size(self):
        return self.N_dis, self.N_images

    def objective_fct(self, q_vlt, predictions, gt_vlts):
        indices = np.argmin((predictions - q_vlt)**2, axis=1)
        predicted_vlts = self.distortions[indices].cpu().numpy()
        error = np.sqrt(np.mean((predicted_vlts - gt_vlts) ** 2))
        return error


    def optimize(self, predictions, min, max):
        predictions = predictions.reshape(self.size()).transpose()
        gt_vlt = self.data
        res = minimize_scalar(self.objective_fct, args=(predictions, gt_vlt), bounds=[min, max])
        q_vlt = res.x
        error = res.fun
        indices = np.argmin((predictions - q_vlt) ** 2, axis=1)
        predicted_vlts = self.distortions[indices].cpu().numpy()
        return error, predicted_vlts


    def plot(self, predictions, min, max, metric_name):
        fig = plt.figure(figsize=(5, 4))
        error, predicted_vlts = self.optimize(predictions, min, max)
        plt.plot(self.images.cpu().numpy(), predicted_vlts, label='Metric Prediction')
        plt.plot(self.images.cpu().numpy(), self.data, label='Ground Truth')

        plt.xlabel('Image')
        plt.ylabel('VLT')

        plt.xlim([0, 19])
        plt.ylim([0, 100])

        title = metric_name+' [RMSE='+str(np.round(error, 3))+']'

        plt.legend()
        plt.title(title)

        fig.savefig(os.path.join('figs', 'aliaksei_3', metric_name+'.png'))

        plt.close(fig)

    def error(self, predictions, min, max):
        error, predicted_vlts = self.optimize(predictions, min, max)
        return error



    def get_row_header(self):
        return ['Distortion', 'Image']

    def get_rows_conditions(self):
        return [self.distortions[self.dis_indices], self.images[self.images_indices]]

    def get_preview_folder(self):
        return 'aliaksei_3'

    def short_name(self):
        return 'Lossless Dataset - Test'