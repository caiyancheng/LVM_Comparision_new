import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Modified_Score_Results_3.csv')

# Left plot data (2x2 grid)
Y_Alignment_score_dimensions_left = ['Phase-Coherent Masking', 'Contrast Matching']
X_dimensions_left = ['DINO ImageNet k-nn Accuracy', 'DINO ImageNet linear Accuracy']
test_models_left = ['DINO', 'DINOv2']

colors_classes_left = [
    "r",  # 红色
    "g",  # 绿色
]

# Right plot data (1x2 grid)
Y_Alignment_score_dimensions_right = ['Contrast Matching']
X_dimensions_right = ['OpenCLIP 38 datasets Average Accuracy', 'OpenCLIP ImageNet 1k Accuracy']
test_models_right = ['OpenCLIP']

colors_classes_right = [
    "b",
]

# Create a larger grid: 3 rows and 2 columns (2x2 left, 1x2 right)
fig, axes = plt.subplots(3, 2, figsize=(9, 6), dpi=300)

# Left plots (2x2 grid) - Top two rows
for i, X_dimension in enumerate(X_dimensions_left):
    for j, Y_dimension in enumerate(Y_Alignment_score_dimensions_left):
        if (i == 0 and j == 1):  # Swap the first row, second column with second row, first column
            ax = axes[1, 0]  # Assign to the second row, first column
        elif (i == 1 and j == 0):
            ax = axes[0, 1]  # Assign to the first row, second column
        else:
            ax = axes[i, j]  # Default assignment for other positions
        X_all_list = []
        Y_all_list = []

        for test_model_index in range(len(test_models_left)):
            test_model = test_models_left[test_model_index]
            subdf = df[df['Models'] == test_model]
            X_list = subdf[X_dimension].tolist()
            Y_list = subdf[Y_dimension].tolist()
            X_all_list = X_all_list + X_list
            Y_all_list = Y_all_list + Y_list
            ax.scatter(X_list, Y_list, c=colors_classes_left[test_model_index], label=test_model)

        pearson_corr = np.corrcoef(X_all_list, Y_all_list)[0, 1]
        corr_text = (f"Pearson Correlation: {pearson_corr:.2f}")
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=0), zorder=10)
        ax.set_xlabel(X_dimension + '(%)', fontsize=12)
        # ax.set_xlabel(X_dimension + ' (Classification)', fontsize=12)
        if Y_dimension == 'Phase-Coherent Masking':
            ax.set_ylabel('Phase-Coherent\nMasking($r_s$ ↑)', fontsize=12)
        if Y_dimension == 'Contrast Matching':
            ax.set_ylabel('Contrast Matching\n(RMSE ↓)', fontsize=12)
        ax.legend()

# Right plots (1x2 grid) - Last row
for i, X_dimension in enumerate(X_dimensions_right):
    for j, Y_dimension in enumerate(Y_Alignment_score_dimensions_right):
        ax = axes[2, i]  # Use the last row for right plots
        X_all_list = []
        Y_all_list = []

        for test_model_index in range(len(test_models_right)):
            test_model = test_models_right[test_model_index]
            subdf = df[df['Models'] == test_model]
            X_list = subdf[X_dimension].tolist()
            Y_list = subdf[Y_dimension].tolist()
            X_all_list = X_all_list + X_list
            Y_all_list = Y_all_list + Y_list
            ax.scatter(X_list, Y_list, c=colors_classes_right[test_model_index], label=test_model)

        pearson_corr = np.corrcoef(X_all_list, Y_all_list)[0, 1]
        corr_text = (f"Pearson Correlation: {pearson_corr:.2f}")
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=0), zorder=10)
        ax.set_xlabel(X_dimension + '(%)', fontsize=12)
        # ax.set_xlabel(X_dimension + ' (Classification)', fontsize=12)
        if Y_dimension == 'Phase-Coherent Masking':
            ax.set_ylabel('Phase-Coherent\nMasking ($r_s$ ↑)', fontsize=12)
        if Y_dimension == 'Contrast Matching':
            ax.set_ylabel('Contrast Matching\n(RMSE ↓)', fontsize=12)
        ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig('combined_plot_3.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.08)
# plt.show()
