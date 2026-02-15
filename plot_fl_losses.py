import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter


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
    epochs = range(1, len(df) + 1)
    
    # Pastel colors for clients
    client_colors = ['#6FA8DC', '#93C47D', '#FFD966']
    
    plt.figure()
    
    # Plot individual clients
    for i in range(1, 4):
        col_name = f"client{i}_loss"
        if col_name in df.columns:
            plt.plot(epochs, df[col_name], color=client_colors[i-1], linewidth=1.5, label=f'Client {i}')
    
    # Plot average as red dotted line
    plt.plot(epochs, df["train_loss"], color='red', linestyle=':', linewidth=2.5, label='Average')
    
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.xlim(0, 50)
    plt.ylim(0.0, 0.15)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--out_dir', required=True, help='Directory to save output figures')
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_file in csv_dir.glob('*.csv'):
        output_file = out_dir / f"{csv_file.stem}.png"
        plot_training_loss(csv_file, output_file)
        print(f"Generated: {output_file}")
