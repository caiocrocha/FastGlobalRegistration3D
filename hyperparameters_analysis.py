import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Load CSV file
    data = pd.read_csv('results_hyperparameters.csv')

    # Extract noise level and test number from the filename
    idx = [x.split("_rot_")[0].split("_")[-2:] for x in data['filename']]
    idx = [(int(x[0]), int(x[1])) if len(x[0]) == 2 else (0, int(x[1])) for x in idx]

    # Add extracted values to DataFrame
    data[['noise', 'test']] = pd.DataFrame(idx, index=data.index)

    # Create subplots for each noise level
    unique_noise_levels = sorted(data['noise'].unique())
    # fig, axes = plt.subplots(1, len(unique_noise_levels), figsize=(15, 5), sharey=True)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    colors = ('green', 'blue', 'red')
    lines = []

    for i, noise_level in enumerate(unique_noise_levels):
        # ax = axes[i]
        subset = data[data['noise'] == noise_level]
        cmap = sns.light_palette(colors[i], as_cmap=True)
        palette = [cmap(i) for i in np.linspace(0.3, 1.0, len(subset['division_factor'].unique()))]
        line = sns.lineplot(data=subset, x='maximum_correspondence_distance', y='rmse', hue='division_factor', marker='o', ax=ax, palette=palette)
        lines.append(line)
        ax.set_xscale('log')
        ax.set_xlabel('Max Correspondence Distance')
        ax.set_ylabel('RMSE') if i == 0 else ax.set_ylabel('')
        # ax.set_title(f'Noise Level: {noise_level}')
        ax.grid(True)

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(data['division_factor'].unique())
    custom_labels = [f"Noise {noise_level}, Div {label}" for noise_level in unique_noise_levels for label in unique_labels]
    ax.legend(handles, custom_labels)

    fig.suptitle('Impact of Maximum Correspondence Distance on RMSE', y=0.95)
    ax.set_title('by Division Factor and Noise Level')
    fig.tight_layout()
    fig.savefig('maximal_correspondence_distance.png')
    fig.show()
