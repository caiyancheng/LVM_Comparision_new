from pycvvdp.vq_metric import vq_metric
from pycvvdp.video_source import *


class VQM(vq_metric):

    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[]):
        if display_photometry is None:
            self.display_photometry = vvdp_display_photometry.load(display_name, config_paths)
            self.display_name = display_name
        else:
            self.display_photometry = display_photometry
            self.display_name = "unspecified"

        if display_geometry is None:
            self.display_photometry = vvdp_display_geometry.load(display_name, config_paths)
        else:
            self.display_geometry = display_geometry
