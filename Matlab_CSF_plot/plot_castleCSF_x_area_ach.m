clear all;
clc;
CSF_model = CSF_castleCSF();

spatial_frequency = 8;
luminance = 100;
luminance_delta = [0.917470668581922, 0.396967232858222, 0.025779610717139];
area_list = logspace(log10(pi*0.01^2),log10(pi*1^2), 100);
% area_list = logspace(log10(0.01),log10(10), 100);

x_ticks = [0.01, 0.1, 1, 10];
y_ticks = [1, 10, 100, 1000];
% ha = tight_subplot(1, 1, [.05 .02], [.1 .07], [.1 .01]);
ha = tight_subplot(1, 1, [.05 .02], [.15 .1], [.15 .05]);

csf_pars = struct('s_frequency', spatial_frequency, 't_frequency', 0, 'orientation', 0, ...
    'luminance', luminance, 'eccentricity', 0, 'area', area_list', 'lms_delta', luminance_delta);
sensitivity_list = CSF_model.sensitivity(csf_pars);
plot(area_list, sensitivity_list, 'LineWidth', 4);

font_size = 8;
set(gca, 'XScale', 'log', 'XTick', x_ticks, 'XTickLabel', x_ticks, 'FontSize', font_size);
set(gca, 'YScale', 'log', 'YTick', y_ticks, 'YTickLabel', y_ticks, 'FontSize', font_size);
xlim([min(area_list), max(area_list)]);
ylim([min(y_ticks), max(y_ticks)]);
xlabel('Stimulus Area (degree^2)', 'FontSize', font_size);
ylabel('Sensitivity', 'FontSize', font_size);
% title(['Luminance = ' num2str(luminance) ' nits, S. Freq = ' num2str(spatial_frequency) ' cpd']);

data = struct('area_list', area_list, 'sensitivity_list', sensitivity_list);
jsonStr = jsonencode(data);
fileID = fopen('castleCSF_area_sensitivity_data.json', 'w');
if fileID == -1
    error('Cannot open file for writing.');
end
fprintf(fileID, '%s', jsonStr);
fclose(fileID);