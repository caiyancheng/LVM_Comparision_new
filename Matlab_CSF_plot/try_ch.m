clear all;
clc;
CSF_model = CSF_castleCSF();
% CSF_model.plot_mechanism('col_mech');
LMS_bkg = [36.75726, 15.81412, 1.041002];
LMS_delta = [0, 0, 0.0196126501593528];
LMS_delta_normalized = LMS_delta / norm(LMS_delta);
csf_pars = struct('s_frequency', 2, 't_frequency', 0, 'orientation', 0, ...
    'lms_bkg', LMS_bkg, 'lms_delta', LMS_delta_normalized,'eccentricity', 0, 'area', 10);
S = CSF_model.sensitivity(csf_pars);
% X = 1;
% % LMS_delta_only_ach = [0.697994684706451, 0.302005315293549, 0.0196126501593528];
% % LMS_delta_RG = [0.302005315293549, -0.302005315293549, 0];
% % LMS_delta_YV = [0, 0, 0.0196126501593528];
% % LMS_delta_only_ach_norm = [0.917470668581922, 0.396967232858222, 0.025779610717139];
% % LMS_delta_RG_norm = [0.707106781186548, -0.707106781186548, 0];
% % LMS_delta_YV_norm = [0, 0, 1];