import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

df = pd.read_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Modified_Score_Results_3.csv')
Y_Alignment_score_dimensions = ['Phase-Coherent Masking', 'Contrast Matching', 'Contrast Matching']
X_dimensions = ['DINO ImageNet k-nn Accuracy', 'DINO ImageNet linear Accuracy']
test_models = ['DINO', 'DINOv2', 'OpenCLIP']

colors_classes = [
    "#C6B74C",  # 暗黄色
    "#E0A883",  # 暗橙色
    "#D68A9A",  # 暗粉色
]

# 创建子图的网格，行数对应X_dimensions，列数对应Y_Alignment_score_dimensions
fig, axes = plt.subplots(len(X_dimensions), len(Y_Alignment_score_dimensions), figsize=(len(Y_Alignment_score_dimensions) * 4, len(X_dimensions) * 4))

# 迭代每个X和Y的维度，并在对应的子图上绘制数据
for i, X_dimension in enumerate(X_dimensions):
    for j, Y_dimension in enumerate(Y_Alignment_score_dimensions):
        ax = axes[i, j]
        X_all_list = []
        Y_all_list = []

        for test_model_index in range(len(test_models)):
            test_model = test_models[test_model_index]
            if not X_dimension.startswith(test_model):
                continue
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
