from pycvvdp import cvvdp
import torch

def load_ckpt(ckpt_path, net):
    # Load network weights
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['state_dict'])
    return net

"""
ColourVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 
"""
class cvvdp_nn(cvvdp):
    input_dims_masking = 6      # T, R, S, T*S, R*S, abs(T-R)*S
    input_dims_pooling = 36     # 9 bands x 4 bands per channel
    rho_dims = 1                # Condition on base rho band

    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False,
                 hidden_dims=8, num_layers=2, dropout=0.2, masking='base', pooling='base', calibrated_ckpt=None, config_paths=[]):
        from torchvision.ops import MLP
        assert masking in ('base', 'mlp')
        self.masking = masking
        if masking == 'mlp':
            # Separate args (hidden_dims, dropout, num_layers, etc) for masking net
            hidden_dims = 16
            num_layers = 4
            self.masking_net = MLP(self.input_dims_masking, [hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout)

        assert pooling in ('base', 'lstm', 'gru')
        self.pooling = pooling
        rnn_type = {'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU}
        if pooling in ('lstm', 'gru'):
            recurrent_net = rnn_type[pooling](self.input_dims_pooling, hidden_dims, num_layers, dropout=dropout)
            linear = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims + self.rho_dims, 1),   # rho_band appended here
                torch.nn.Sigmoid()
            )
            self.pooling_net = torch.nn.Sequential(recurrent_net, linear)

        super().__init__(display_name=display_name, display_photometry=display_photometry,
                         display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
                         quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
                         calibrated_ckpt=calibrated_ckpt)

        if masking == 'mlp':
            self.masking_net.to(self.device)
            self.masking_net.eval()

        if pooling in ('lstm', 'gru'):
            self.pooling_net.to(self.device)
            self.pooling_net.eval()

    def update_from_checkpoint(self, ckpt):
        super().update_from_checkpoint(ckpt)
        if self.masking == 'mlp':
            prefix = 'masking_net.'
            if torch.cuda.is_available():
                state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt)['state_dict'].items() if key.startswith(prefix)}
            else:
                state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt, map_location=torch.device('cpu'))['state_dict'].items() if key.startswith(prefix)}
            self.masking_net.load_state_dict(state_dict)
        if self.pooling in ('lstm', 'gru'):
            prefix = 'pooling_net.'
            if torch.cuda.is_available:
                state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt)['state_dict'].items() if key.startswith(prefix)}
            else:
                state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt, map_location=torch.device('cpu'))['state_dict'].items() if key.startswith(prefix)}
            self.pooling_net.load_state_dict(state_dict)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, features_provided=False):
        if not features_provided:
            return super().predict_video_source(vid_source)
        else:
            Q_jod = self.do_pooling_and_jods(*vid_source)

            stats = {}
            return (Q_jod.squeeze(), stats)

    def apply_masking_model(self, T, R, S):
        if self.masking == 'mlp':
            c, n, h, w = T.shape
            if S.dim() == 0:
                S = torch.full_like(T, S)
            #D_base = super().apply_masking_model(T, R, S)   # v2
            T, R, S = T.flatten(), R.flatten(), S.flatten()
            feat_in = torch.stack((T, R, S, T*S, R*S, torch.abs(T - R)*S), dim=-1)
            batch_size = 2560*1440     # Split larger than 2k into multiple batches
            mlp_out = torch.cat([self.masking_net(batch) for batch in feat_in.split(batch_size)]).squeeze(-1)
            # D = mlp_out.reshape(c, n, h, w)     # v1
            # D = D_base * mlp_out.reshape(c, n, h, w)        # v2
            D = ((S*torch.abs(T - R)**self.mask_p) /                                # v3
                (1 + torch.nn.functional.softplus(mlp_out))).reshape(c, n, h, w)    # v3
        else:
            D = super().apply_masking_model(T, R, S)
        return D

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, Q_per_ch, base_rho_band, fps):
        if self.pooling == 'base':
            return super().do_pooling_and_jods(Q_per_ch, base_rho_band, fps)
        # Q_per_ch[channel,frame,sp_band]
        feat_in = Q_per_ch.permute(1, 0, 2).flatten(start_dim=1)
        feat_intermediate, _ = self.pooling_net[0](feat_in)
        feat_intermediate = torch.cat((feat_intermediate[-1], torch.as_tensor(base_rho_band, device=self.device, dtype=torch.float32).unsqueeze(0)))
        Q = self.pooling_net[1](feat_intermediate).squeeze() * 10
        return Q

    def short_name(self):
        return f"cvvdp_mask-{self.masking}_pool-{self.pooling}"

    def quality_unit(self):
        return "JOD"

    def get_info_string(self):
        if self.display_name.startswith('standard_'):
            #append this if are using one of the standard displays
            standard_str = ', (' + self.display_name + ')'
        else:
            standard_str = ''
        fv_mode = 'foveated' if self.foveated else 'non-foveated'
        return '"ColourVideoVDP with {} v{}, {:.4g} [pix/deg], Lpeak={:.5g}, Lblack={:.4g} [cd/m^2], {}{}"'.format(self.pooling, self.version, self.pix_per_deg, self.display_photometry.get_peak_luminance(), self.display_photometry.get_black_level(), fv_mode, standard_str)

    def train(self):
        if self.masking == 'mlp':
            self.masking_net.train()
        if self.pooling in ('lstm', 'gru'):
            self.pooling_net.train()

    def eval(self):
        if self.masking == 'mlp':
            self.masking_net.eval()
        if self.pooling in ('lstm', 'gru'):
            self.pooling_net.eval()
