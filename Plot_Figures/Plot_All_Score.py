import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('Score_Results_2_new.csv')
plot_dimensions = ['Spatial Frequency - Gabor Achromatic', 'Spatial Frequency - Noise Achromatic',
                   'Spatial Frequency - Gabor RG', 'Spatial Frequency - Gabor YV', 'Luminance', 'Area',
                   'Phase-Coherent Masking', 'Phase-Incoherent Masking', 'Contrast Matching']
add_tasks = ['Contrast Detection', 'Contrast Detection', 'Contrast Detection', 'Contrast Detection',
             'Contrast Detection', 'Contrast Detection', 'Contrast Masking', 'Contrast Masking',
             'Supra-threshold Vision']

# 获取模型相关信息
Models = df['Models'].tolist()
Architecture = df['Architecture'].tolist()
Train_Dataset = df['Training dataset'].tolist()
Model_classes = ['No Encoder', 'DINO', 'DINOv2', 'OpenCLIP', 'SAM', 'SAM-2', 'MAE', 'SD-VAE', 'LPIPS', 'ST-LPIPS', 'ColorVideoVDP']
colors_classes = [
    "#D3D3D3",  # 浅灰色
    "#F0E68C",  # 柔和的黄色
    "#FFDAB9",  # 柔和的橙色
    "#FFC0CB",  # 柔和的粉色
    "#D8BFD8",  # 柔和的紫色
    "#ADD8E6",  # 柔和的蓝色
    "#98FB98",  # 柔和的绿色
    "#AFEEEE",  # 柔和的青色
    "#778877",  # 灰绿色
    "#708090",  # 灰蓝色
    "#BC8F8F"   # 灰红色
]

# 处理颜色和模型全称信息
color_list = []
plot_model_full_name = []
for model_index, model in enumerate(Models):
    index_model_class = Model_classes.index(model)
    plot_color = colors_classes[index_model_class]
    color_list.append(plot_color)
    if model == 'No Encoder' or model == 'ColorVideoVDP':
        model_full_name = f'{Models[model_index]}'
    else:
        model_full_name = f'{Models[model_index]} - {Architecture[model_index]} - {Train_Dataset[model_index]}'
    plot_model_full_name.append(model_full_name)

# 创建子图
# fig, axes = plt.subplots(len(plot_dimensions), 1, figsize=(11, len(plot_dimensions) * 2.1))
fig, axes = plt.subplots(len(plot_dimensions), 1, figsize=(25, len(plot_dimensions) * 2.1))

# 绘制每个维度的直方图
for i, (dimension, adt, ax) in enumerate(zip(plot_dimensions, add_tasks, axes)):
    score_values = df[dimension].tolist()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bars = ax.bar(np.arange(0, len(score_values)), score_values, color=color_list, width=0.8)

    for bar, value in zip(bars, score_values):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.01, f"{value:.2f}", ha='center',
                va='bottom', fontsize=10, rotation=90)
    # ax.set_ylabel("$\\rho$ - " + dimension.replace("Spatial Frequency", "Sp. Freq."), fontsize=14)
    if i == len(plot_dimensions) - 1:
        ax.set_ylabel("RMSE", fontsize=14)
        title_text = adt + ' - ' + dimension + ' - Root Mean Squared Error ↓'
        ax.set_ylim(0, max(score_values))
    else:
        ax.set_ylabel("Spearman $r_s$", fontsize=14)
        title_text = adt + ' - ' + dimension + ' - Spearman Correlation ↑'
        ax.set_ylim(0, 1)
    ax.text(0.5, 1.5, title_text, ha='center', va='bottom', fontsize=14.5, transform=ax.transAxes, fontweight='bold')

    # 设置 x 轴的显示范围，以消除左右的空白
    ax.set_xlim(-0.5, len(score_values) - 0.5)

    # 在最后一个子图上添加模型名称
    if i == len(plot_dimensions) - 1:
        ax.set_xticks(np.arange(len(plot_model_full_name)))
        ax.set_xticklabels(plot_model_full_name, rotation=90, fontsize=12, ha='center')
    else:
        ax.set_xticks([])

# 调整图形布局以减少空白区域
plt.subplots_adjust(hspace=0.8, right=1)
plt.tight_layout(pad=0.2)
plt.savefig('All_Score_Result.png', format='png', dpi=300, bbox_inches='tight')
# plt.savefig('All_Score_Result_4.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.show()