import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Score_Results_3.csv')
Alignment_score_dimensions = ['Spatial Frequency - Gabor Achromatic', 'Spatial Frequency - Noise Achromatic',
                   'Spatial Frequency - Gabor RG', 'Spatial Frequency - Gabor YV', 'Luminance', 'Area',
                   'Phase-Coherent Masking', 'Phase-Incoherent Masking', 'Contrast Matching']
df['Contrast Masking Average'] = df[['Phase-Coherent Masking', 'Phase-Incoherent Masking']].mean(axis=1)
df['Contrast Detection Average'] = df[['Spatial Frequency - Gabor Achromatic', 'Spatial Frequency - Noise Achromatic',
                   'Spatial Frequency - Gabor RG', 'Spatial Frequency - Gabor YV', 'Luminance', 'Area']].mean(axis=1)
df['Masking Matching Sum'] = (df[['Phase-Coherent Masking', 'Phase-Incoherent Masking']].sum(axis=1) - df['Contrast Matching'])
df['Overall Score Sum'] = (df[['Spatial Frequency - Gabor Achromatic', 'Spatial Frequency - Noise Achromatic',
                                'Spatial Frequency - Gabor RG', 'Spatial Frequency - Gabor YV', 'Luminance', 'Area',
                                'Phase-Coherent Masking', 'Phase-Incoherent Masking']].sum(axis=1) - df['Contrast Matching'])
df.to_csv(r'E:\Py_codes\LVM_Comparision\Plot_Figures/Modified_Score_Results_3.csv', index=False)