import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys

# Function to ensure the file path is valid
def verify_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")
    return path

# Retrieve index from command line argument if available
try:
    index = sys.argv[1]
except IndexError:
    print("Usage: python script.py <index>")
    sys.exit(1)

# Define paths
in_file_path = verify_file(f'/users/home/parentfolder/Augment/result/in_shadow_observations_{index}.csv')
out_file_path = verify_file(f'/users/home/parentfolder/Augment/result/out_shadow_observations_{index}.csv')
output_file_path = f'/users/home/parentfolder/Augment/result/gaussian_params_{index}.csv'

# Load the CSV files
in_data = pd.read_csv(in_file_path)['0']
out_data = pd.read_csv(out_file_path)['0']

# Fit the data to normal distributions
in_mean, in_std = norm.fit(in_data)
out_mean, out_std = norm.fit(out_data)

# Create a DataFrame to hold the Gaussian parameters
gaussian_params = pd.DataFrame({
    'Model': ['In', 'Out'],
    'Mean': [in_mean, out_mean],
    'Standard Deviation': [in_std, out_std],
    'Index': [index, index]  # Add index to the dataframe
})

# Save the Gaussian parameters to a CSV file
gaussian_params.to_csv(output_file_path, index=False)

# Print to verify
print(gaussian_params)
