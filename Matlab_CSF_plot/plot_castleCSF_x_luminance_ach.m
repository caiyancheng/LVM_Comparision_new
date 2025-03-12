clear all;
clc;
CSF_model = CSF_castleCSF();

spatial_frequency = 2;
luminance_list = logspace(log10(0.1),log10(200), 100);
luminance_delta = [0.917470668581922, 0.396967232858222, 0.025779610717139];
area = pi*1^2;

x_ticks = [0.01, 0.1, 1, 10, 100];
y_ticks = [1, 10, 100, 1000];
% ha = tight_subplot(1, 1, [.05 .02], [.1 .07], [.1 .01]);
ha = tight_subplot(1, 1, [.05 .02], [.15 .1], [.15 .05]);

csf_pars = struct('s_frequency', spatial_frequency, 't_frequency', 0, 'orientation', 0, ...
    'luminance', luminance_list', 'eccentricity', 0, 'area', area, 'lms_delta', luminance_delta);
sensitivity_list = CSF_model.sensitivity(csf_pars);
plot(luminance_list, sensitivity_list, 'LineWidth', 4);

font_size = 8;
set(gca, 'XScale', 'log', 'XTick', x_ticks, 'XTickLabel', x_ticks, 'FontSize', font_size);
set(gca, 'YScale', 'log', 'YTick', y_ticks, 'YTickLabel', y_ticks, 'FontSize', font_size);
xlim([min(luminance_list), max(luminance_list)]);
ylim([min(y_ticks), max(y_ticks)]);
xlabel('Luminance (nits)', 'FontSize', font_size);
ylabel('Sensitivity', 'FontSize', font_size);

data = struct('luminance_list', luminance_list, 'sensitivity_list', sensitivity_list);
jsonStr = jsonencode(data);
fileID = fopen('castleCSF_luminance_sensitivity_data.json', 'w');
if fileID == -1
    error('Cannot open file for writing.');
end
fprintf(fileID, '%s', jsonStr);
fclose(fileID);
