import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Modified_Score_Results_3.csv')
Y_Alignment_score_dimensions = ['Contrast Masking Average', 'Contrast Detection Average', 'Phase-Coherent Masking',
                                'Contrast Matching', 'Masking Matching Sum', 'Overall Score Sum']
X_dimensions = ['Params_All_MB', 'Params_Forward_MB', 'GFlops']
test_models = ['DINO', 'DINOv2', 'OpenCLIP', 'SAM', 'SAM-2', 'MAE']

colors_classes = [
    "#F0E68C",  # 柔和的黄色
    "#FFDAB9",  # 柔和的橙色
    "#FFC0CB",  # 柔和的粉色
    "#D8BFD8",  # 柔和的紫色
    "#ADD8E6",  # 柔和的蓝色
    "#98FB98",  # 柔和的绿色
]

# 创建子图的网格，行数对应X_dimensions，列数对应Y_Alignment_score_dimensions
fig, axes = plt.subplots(len(X_dimensions), len(Y_Alignment_score_dimensions), figsize=(30, 15))

# 迭代每个X和Y的维度，并在对应的子图上绘制数据
for i, X_dimension in enumerate(X_dimensions):
    for j, Y_dimension in enumerate(Y_Alignment_score_dimensions):
        ax = axes[i, j]

        for test_model_index in range(len(test_models)):
            test_model = test_models[test_model_index]
            subdf = df[df['Models'] == test_model]
            X_list = subdf[X_dimension].tolist()
            Y_list = subdf[Y_dimension].tolist()
            ax.scatter(X_list, Y_list, c=colors_classes[test_model_index], label=test_model)

        ax.set_xlabel(X_dimension)
        ax.set_ylabel(Y_dimension)
        ax.set_xscale('log')
        ax.legend()

# 调整子图布局
plt.tight_layout()
plt.show()
