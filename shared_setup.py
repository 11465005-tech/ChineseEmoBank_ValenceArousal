import os

import matplotlib
import torch

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def setup_runtime():
    """Provide a lightweight runtime configuration for the ChineseEmoBank pipeline.
    This keeps the notebook/script imports stable across local and Colab-like hosts.
    """
    font = "DejaVu Sans"
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            font = "DejaVu Sans"
            break

    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return font, device
