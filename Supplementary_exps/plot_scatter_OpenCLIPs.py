import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

df = pd.read_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Modified_Score_Results_3.csv')
Y_Alignment_score_dimensions = ['Contrast Matching']
X_dimensions = ['OpenCLIP 38 datasets Average Accuracy', 'OpenCLIP ImageNet 1k Accuracy']
test_models = ['OpenCLIP']

colors_classes = [
    "#D68A9A",
]

# 创建子图的网格，行数对应X_dimensions，列数对应Y_Alignment_score_dimensions
fig, axes = plt.subplots(len(X_dimensions), len(Y_Alignment_score_dimensions), figsize=(len(Y_Alignment_score_dimensions) * 4, len(X_dimensions) * 4))

if len(X_dimensions) == 1:
    axes = np.expand_dims(axes, axis=0)
if len(Y_Alignment_score_dimensions) == 1:
    axes = np.expand_dims(axes, axis=1)

# 迭代每个X和Y的维度，并在对应的子图上绘制数据
for i, X_dimension in enumerate(X_dimensions):
    for j, Y_dimension in enumerate(Y_Alignment_score_dimensions):
        ax = axes[i, j]
        X_all_list = []
        Y_all_list = []

        for test_model_index in range(len(test_models)):
            test_model = test_models[test_model_index]
            subdf = df[df['Models'] == test_model]
            X_list = subdf[X_dimension].tolist()
            Y_list = subdf[Y_dimension].tolist()
            X_all_list = X_all_list + X_list
            Y_all_list = Y_all_list + Y_list
            ax.scatter(X_list, Y_list, c=colors_classes[test_model_index], label=test_model)

        pearson_corr = np.corrcoef(X_all_list, Y_all_list)[0, 1]
        corr_text = (f"Pearson Correlation: {pearson_corr:.2f}")
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='none', alpha=0), zorder=10)
        ax.set_xlabel(X_dimension)
        ax.set_ylabel(Y_dimension)
        ax.legend()

# 调整子图布局
plt.tight_layout()
plt.show()
