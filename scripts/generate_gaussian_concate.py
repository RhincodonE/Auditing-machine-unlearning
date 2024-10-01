import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import re

# Define the directory containing the results
result_dir = '/users/home/parentfolder/Augment/result/'

# Get all files in the result directory
files = os.listdir(result_dir)

# Helper function to extract index from filename
def extract_index(filename):
    match = re.search(r'(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None

# Filter and group in and out observation files by their indices
in_files = {extract_index(f): f for f in files if "in_observations" in f}
out_files = {extract_index(f): f for f in files if "out_observations" in f}

# Ensure matching indices
assert in_files.keys() == out_files.keys(), "Mismatch in indices of in and out files"

# Colors for each index
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#2ca02c', '#d62728']

# Create a figure and axis
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Process each pair of matched in and out files
for idx, (index) in enumerate(sorted(in_files.keys())):
    color = colors[idx % len(colors)]  # Cycle through colors

    in_file = in_files[index]
    out_file = out_files[index]

    # Load the data from files
    in_data = pd.read_csv(os.path.join(result_dir, in_file))['0']
    out_data = pd.read_csv(os.path.join(result_dir, out_file))['0']

    # Fit the data to normal distributions
    in_mean, in_std = norm.fit(in_data)
    out_mean, out_std = norm.fit(out_data)

    # Generate points for the probability density functions
    x_in = np.linspace(in_data.min(), in_data.max(), 100)
    x_out = np.linspace(out_data.min(), out_data.max(), 100)
    p_in = norm.pdf(x_in, in_mean, in_std)
    p_out = norm.pdf(x_out, out_mean, out_std)

    # Plot the distributions
    ax.plot(x_in, p_in, linestyle='-', color=color, label=f'In-Index {index} (μ={in_mean:.2f}, σ={in_std:.2f})')
    ax.plot(x_out, p_out, linestyle='dotted', color=color, label=f'Out-Index {index} (μ={out_mean:.2f}, σ={out_std:.2f})')

# Set the plot title and labels
plt.title('Comparison of Normal Distributions of In and Out Observations')
plt.xlabel('Logits')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(os.path.join(result_dir, 'combined_distribution.png'))
plt.show()
