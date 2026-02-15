import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
# Global plot style (set once)
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 12,
    "axes.labelsize": 14,      
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
})
def plot_training_loss(csv_path, output_path):
    df = pd.read_csv(csv_path)
    # Epochs: start at 1, but show 0 on x-axis
    epochs = range(1, len(df) + 1)
    loss = df["training_loss"]
    plt.figure()
    plt.plot(epochs, loss)
    # Axis labels
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    # Fixed axis limits
    plt.xlim(0, 100)     
    plt.ylim(0.0, 0.25)
    # Force y-axis to show max 2 decimals
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Grid
    plt.grid(True)
    # Layout & save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
if __name__ == "__main__":
    import sys
    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    plot_training_loss(csv_path, output_path)
