import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history, to_forecast):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    plt.show()
    # plt.savefig(f'images/loss_curves/{model_name}_{to_forecast}.png')

def plot_residuals_distribution(y_true, y_pred, to_forecast):
    errors = y_pred - y_true
    errors_plot = errors[
        [
            f"{to_forecast}_future1",
            f"{to_forecast}_future10",
            f"{to_forecast}_future20",
            f"{to_forecast}_future30",
            f"{to_forecast}_future40",
        ]
    ]
    errors_plot = np.abs(errors_plot)

    # rename columns for plotting
    errors_plot.columns = ["1h", "10h", "20h", "30h", "40h"]

    sns.violinplot(errors_plot)  # , hue="alive")
    plt.show()
    return errors