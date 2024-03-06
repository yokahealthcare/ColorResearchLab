import sys

import numpy as np
import pandas as pd
from skimage import color

from rich.console import Console
from rich.theme import Theme

console = Console()

import numpy as np


def manhattan_distance(a, b):
    """Calculates the Manhattan distance between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.

    Returns:
        float: The Manhattan distance between the vectors.

    Raises:
        ValueError: If the input vectors have different shapes.
    """

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.sum(np.abs(a - b))


def chebyshev_distance(a, b):
    """Calculates the Chebyshev distance between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.

    Returns:
        float: The Chebyshev distance between the vectors.

    Raises:
        ValueError: If the input vectors have different shapes.
    """

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")
    return np.max(np.abs(a - b))


def minkowski_distance(a, b, p):
    """Calculates the Minkowski distance between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.
        p (float): The power parameter (must be greater than 0).

    Returns:
        float: The Minkowski distance between the vectors.

    Raises:
        ValueError: If the input vectors have different shapes or p is not a positive value.
    """

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")
    if p <= 0:
        raise ValueError("p must be a positive value.")
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


def euclidean_distance(a, b, axis=None):
    """Calculates the Euclidean distance between vectors or the corresponding rows or columns in matrices.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array, with the same shape as `a` or compatible broadcasting.
        axis (int, optional): The axis along which to compute the distance.
            - None (default): Computes the distance between vectors or entire matrices based on their shapes.
            - 0: Computes the distance between rows.
            - 1: Computes the distance between columns.

    Returns:
        np.ndarray: An array containing the Euclidean distances.

    Raises:
        ValueError: If the input arrays have incompatible shapes for broadcasting except for the specified axis.
    """

    if len(a.shape) != len(b.shape) and any(d != 1 for d in np.subtract(a.shape, b.shape) if d is not None):
        raise ValueError("Input arrays must have compatible shapes, except for the specified axis.")

    return np.linalg.norm(a - b, axis=axis)


def predict_color(rgb, iscc_lab, iscc_color, iscc_category, p):
    rgb = np.array(rgb)
    rgb = rgb / 255
    lab = color.rgb2lab(rgb)

    # Find the closest color in LUT1
    t = []
    for i in range(len(iscc_lab)):
        distances = minkowski_distance(iscc_lab[i], lab, p)
        t.append(distances)
    distances = np.array(t)
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

correct = 0
wrong = 0
for p in range(1, 500):
    for idx, i in enumerate(testing_rgb):
        custom_theme = Theme({"custom": f"rgb({i[0]},{i[1]},{i[2]}) bold"})
        console = Console(theme=custom_theme)

        p_col, p_cat = predict_color(i, iscc_lab, iscc_color, iscc_category, p)
        if p_cat in testing_color[idx][0]:
            # console.print(f"[bold green][CORRECT][/bold green] " + f"P : {p_cat} ({p_col});".ljust(
            #     60) + f"R : {testing_color[idx][0]} {i} " + f"[custom]#EXAMPLE[/custom]")
            correct += 1
        else:
            # console.print(f"[bold red][ WRONG ][/bold red] " + f"P : {p_cat} ({p_col});".ljust(
            #     60) + f"R : {testing_color[idx][0]} {i} " + f"[custom]#EXAMPLE[/custom]")
            wrong += 1

    print(f"Percentage of Correct Prediction : {round(correct / len(testing_color) * 100, 2)} %")
    correct = 0
    wrong = 0
