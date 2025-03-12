clear all;
clc;

F = readtable( 'foley_data.csv' );
db2lin = @(c_db) 10.^(c_db/20);
ss = (F.orientation == 0) & (F.c_mask~=0);
x_mask_contrast_list = db2lin(F.c_mask(ss));
y_test_contrast_list = db2lin(F.c_target(ss));
plot(x_mask_contrast_list, y_test_contrast_list, 'o--r' );
set( gca, 'XScale', 'log' );
set( gca, 'YScale', 'log' );

xlabel( 'Mask contrast' );
ylabel( 'Test contrast' );

data = struct('mask_contrast_list', x_mask_contrast_list, 'test_contrast_list', y_test_contrast_list);
jsonStr = jsonencode(data);
fileID = fopen('foley_contrast_masking_data_gabor.json', 'w');
if fileID == -1
    error('Cannot open file for writing.');
end
fprintf(fileID, '%s', jsonStr);
fclose(fileID);
