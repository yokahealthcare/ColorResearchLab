import sys

import numpy as np
import pandas as pd
from skimage import color

from rich.console import Console

console = Console()


def predict_color(rgb, iscc_lab, iscc_color, iscc_category):
    rgb = np.array(rgb)
    rgb = rgb / 255
    lab = color.rgb2lab(rgb)

    # Find the closest color in LUT1
    distances = np.linalg.norm(iscc_lab - lab, axis=1)
    closest_color_index = np.argmin(distances)

    # Get the dominant hue from LUT1
    dominant_hue_color = str(iscc_color[closest_color_index][0])
    dominant_hue_category = str(iscc_category[closest_color_index][0])
    return dominant_hue_color, dominant_hue_category


df = pd.read_excel('asset/data/iscc-nbs-colour-system.xlsx')
df = df.dropna(subset=['r', 'g', 'b'])
df.reset_index(drop=True, inplace=True)

iscc_color = df[["color"]].values
iscc_category = df[["category"]].values
iscc_rgb = df[['r', 'g', 'b']]
iscc_rgb = iscc_rgb.to_numpy()
iscc_rgb = iscc_rgb / 255
iscc_lab = color.rgb2lab(iscc_rgb)

# input_rgb = [100, 200, 200]
# print(f"Input RGB: {input_rgb}")
# predicted_color, predicted_category = predict_color(input_rgb, iscc_lab, iscc_color, iscc_category)
# print(f"Predicted color: {predicted_color}")
# print(f"Predicted category: {predicted_category}")

df_testing = pd.read_csv("asset/data/testing/ral_standard.csv")
target_color = np.unique(iscc_category)
df_testing = df_testing[df_testing["English"].str.contains("|".join(target_color))]
df_testing.reset_index(drop=True, inplace=True)

testing_color = df_testing[["English"]].values
testing_rgb = np.array([i[0].split("-") for i in df_testing[["RGB"]].values]).astype(int)

for idx, i in enumerate(testing_rgb):
    p_col, p_cat = predict_color(i, iscc_lab, iscc_color, iscc_category)
    if p_cat in testing_color[idx][0]:
        console.print(f"[bold green][CORRECT][/bold green] P : {p_cat} ({p_col}); \t\t\t\t R : {testing_color[idx][0]} {i}")
    else:
        console.print(f"[bold red][ WRONG ][/bold red] P : {p_cat} ({p_col}); \t\t\t\t R : {testing_color[idx][0]} {i}")