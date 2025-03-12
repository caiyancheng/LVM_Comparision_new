clear all;
clc;
CSF_model = CSF_castleCSF();

spatial_frequency_list = logspace(log10(0.5),log10(32), 100);
luminance_list = logspace(log10(0.1),log10(1000), 5);
area_list = logspace(log10(0.1),log10(1000), 5);

x_ticks = [0.5, 1,2,4,8,16,32];
y_ticks = [1, 10, 100, 1000];
ha = tight_subplot(length(luminance_list), length(area_list), [.05 .02], [.1 .01], [.05 .01]);
for luminance_index = 1:length(luminance_list)
    luminance_value = luminance_list(luminance_index);
    for area_index = 1:length(area_list)
        area_value = area_list(area_index);
        plot_index = (luminance_index-1) * length(area_list) + area_index;
        axes(ha(plot_index));
        csf_pars = struct('s_frequency', spatial_frequency_list', 't_frequency', 0, 'orientation', 0, ...
            'luminance', luminance_value, 'eccentricity', 0, 'area', area_value);
        sensitivity_list = CSF_model.sensitivity(csf_pars);
        plot(spatial_frequency_list, sensitivity_list);

        set(gca, 'XScale', 'log', 'XTick', x_ticks, 'XTickLabel', x_ticks);
        set(gca, 'YScale', 'log', 'YTick', y_ticks, 'YTickLabel', y_ticks);
        xlim([min(x_ticks), max(x_ticks)]);
        ylim([min(y_ticks), max(y_ticks)]);
        xlabel('Spatial Frequency (cpd)');
        ylabel('Sensitivity');
        title(['Luminance = ' num2str(luminance_value) ' nits, Area = ' num2str(area_value) ' degree^2']);
    end
end

