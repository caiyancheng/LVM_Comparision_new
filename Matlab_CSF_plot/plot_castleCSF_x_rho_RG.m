clear all;
clc;
CSF_model = CSF_castleCSF();

spatial_frequency_list = logspace(log10(0.5),log10(32), 100);
luminance = 100;
luminance_delta = [0.707106781186548, -0.707106781186548, 0];
area = pi*1^2;

x_ticks = [0.5, 1,2,4,8,16,32];
y_ticks = [1, 10, 100, 1000];
% ha = tight_subplot(1, 1, [.05 .02], [.1 .07], [.1 .01]);
ha = tight_subplot(1, 1, [.05 .02], [.15 .1], [.15 .05]);

% csf_pars = struct('s_frequency', spatial_frequency_list', 't_frequency', 0, 'orientation', 0, ...
%     'luminance', luminance, 'eccentricity', 0, 'area', area);
csf_pars = struct('s_frequency', spatial_frequency_list', 't_frequency', 0, 'orientation', 0, ...
    'luminance', luminance, 'lms_delta', luminance_delta,'eccentricity', 0, 'area', area);

sensitivity_list = CSF_model.sensitivity(csf_pars);
plot(spatial_frequency_list, sensitivity_list, 'Color', 'r', 'LineWidth', 4);

font_size = 8;
set(gca, 'XScale', 'log', 'XTick', x_ticks, 'XTickLabel', x_ticks, 'FontSize', font_size);
set(gca, 'YScale', 'log', 'YTick', y_ticks, 'YTickLabel', y_ticks, 'FontSize', font_size);
xlim([min(x_ticks), max(x_ticks)]);
ylim([min(y_ticks), max(y_ticks)]);
xlabel('Spatial Frequency (cpd)', 'FontSize', font_size);
ylabel('Sensitivity', 'FontSize', font_size);

data = struct('rho_list', spatial_frequency_list, 'sensitivity_list', sensitivity_list);
jsonStr = jsonencode(data);
fileID = fopen('castleCSF_rho_sensitivity_data_RG.json', 'w');
if fileID == -1
    error('Cannot open file for writing.');
end
fprintf(fileID, '%s', jsonStr);
fclose(fileID);

