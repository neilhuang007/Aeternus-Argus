import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff


def rendergraph(data, meta, feature1, feature2, output_dir, dataset_name):
    """
    Renders and saves eCDF, Heatmap, and Scatterplots for given features for VPN and Non-VPN data.

    Parameters:
        data (tuple): ARFF data.
        meta (obj): Metadata containing feature names.
        feature1 (str): First feature/column to visualize.
        feature2 (str): Second feature/column to visualize.
        output_dir (str): Directory to save graphs.
        dataset_name (str): Name of the dataset for file naming.
    """
    # Extract column names from metadata
    columns = meta.names()
    df = pd.DataFrame(data, columns=columns)

    # Clean data: replace invalid numeric values with NaN
    df.replace(b'', np.nan, inplace=True)

    # Rename columns for readability
    df.rename(columns={feature1: 'Feature1', feature2: 'Feature2', 'class1': 'Class'}, inplace=True)

    # Separate VPN and Non-VPN users
    vpn_users = df[df['Class'] == b'VPN'].copy()
    non_vpn_users = df[df['Class'] == b'Non-VPN'].copy()

    # Convert relevant columns to numeric
    vpn_users.loc[:, 'Feature1'] = pd.to_numeric(vpn_users['Feature1'], errors='coerce')
    vpn_users.loc[:, 'Feature2'] = pd.to_numeric(vpn_users['Feature2'], errors='coerce')
    non_vpn_users.loc[:, 'Feature1'] = pd.to_numeric(non_vpn_users['Feature1'], errors='coerce')
    non_vpn_users.loc[:, 'Feature2'] = pd.to_numeric(non_vpn_users['Feature2'], errors='coerce')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create eCDF plots
    def plot_ecdf(data, label, ax):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, marker='.', linestyle='none', label=label)
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('ECDF')
        ax.legend()

    fig, ax = plt.subplots()
    plot_ecdf(vpn_users['Feature1'].dropna(), f'VPN Users ({feature1})', ax)
    plot_ecdf(non_vpn_users['Feature1'].dropna(), f'Non-VPN Users ({feature1})', ax)
    plt.title(f'eCDF Plot for {feature1}')
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ecdf.png"))
    plt.close()

    # Create heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(vpn_users[['Feature1', 'Feature2']].corr(), ax=ax[0], cmap='coolwarm', annot=True)
    ax[0].set_title(f'VPN Users Heatmap ({feature1}, {feature2})')
    sns.heatmap(non_vpn_users[['Feature1', 'Feature2']].corr(), ax=ax[1], cmap='coolwarm', annot=True)
    ax[1].set_title(f'Non-VPN Users Heatmap ({feature1}, {feature2})')
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_heatmap.png"))
    plt.close()

    # Create scatter plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(vpn_users['Feature1'], vpn_users['Feature2'], alpha=0.5)
    ax[0].set_title(f'VPN Users Scatter Plot ({feature1} vs {feature2})')
    ax[0].set_xlabel(feature1)
    ax[0].set_ylabel(feature2)
    ax[1].scatter(non_vpn_users['Feature1'], non_vpn_users['Feature2'], alpha=0.5, color='orange')
    ax[1].set_title(f'Non-VPN Users Scatter Plot ({feature1} vs {feature2})')
    ax[1].set_xlabel(feature1)
    ax[1].set_ylabel(feature2)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_scatter.png"))
    plt.close()


# Path and user input
path = 'vpndata/'
output_dir = 'graphs/'
feature1 = 'flowBytesPerSecond'  # Input parameter: feature name for first feature
feature2 = 'flowPktsPerSecond'  # Input parameter: feature name for second feature

# Iterate through all subdirectories and process files
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.arff'):
            file_path = os.path.join(root, file)
            dataset_name = os.path.splitext(file)[0]
            print(f"Processing file: {file_path}")

            # Load ARFF file
            data, meta = arff.loadarff(file_path)

            # Render and save graphs for given features
            rendergraph(data, meta, feature1, feature2, output_dir, dataset_name)
