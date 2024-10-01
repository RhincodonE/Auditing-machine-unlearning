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

# Process each pair of matched in and out files
for index in in_files.keys():
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
    plt.figure(figsize=(8, 6))
    plt.plot(x_in, p_in, 'b-', label=f'In-Model (mean={in_mean:.2f}, std={in_std:.2f})')
    plt.plot(x_out, p_out, 'r-', label=f'Out-Model (mean={out_mean:.2f}, std={out_std:.2f})')
    plt.title(f'Normal Distributions for Index {index}')
    plt.xlabel('Logits')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, f'distribution_index_{index}.png'))
    plt.close()
