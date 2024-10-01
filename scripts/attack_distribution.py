import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys

def load_and_fit_gaussian(file_path):
    """Load observations from a file and fit to a Gaussian distribution."""
    data = pd.read_csv(file_path)['0']
    mean, std = norm.fit(data)
    return mean, std

def determine_model(in_params, out_params):
    """Determine the closest model based on KL divergence."""
    in_mean, in_std = in_params
    out_mean, out_std = out_params
    # Calculate the KL divergence to both models (simplified, assuming equal variance)
    kl_in = np.log(out_std / in_std) + (in_std**2 + (in_mean - out_mean)**2) / (2 * out_std**2) - 0.5

    return kl_in

def process_files(index, results_dir):
    """Process all files for a given index and store results."""
    in_params = (gaussian_params.loc['In', 'Mean'], gaussian_params.loc['In', 'Standard Deviation'])
    out_params = (gaussian_params.loc['Out', 'Mean'], gaussian_params.loc['Out', 'Standard Deviation'])
    results = []

    files_in = [f for f in os.listdir(results_dir) if f.startswith(f'in_observations_{index}_')]
    files_out = [f for f in os.listdir(results_dir) if f.startswith(f'out_observations_{index}_')]

# Sort files by the numeric suffix to ensure proper alignment
    files_in.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files_out.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Printing the sorted lists for verification
# Using zip to iterate over paired elements from both lists
    for file_in, file_out in zip(files_in, files_out):
        file_in_path = os.path.join(results_dir, file_in)
        file_out_path = os.path.join(results_dir, file_out)
        fitted_in_params = load_and_fit_gaussian(file_in_path)
        fitted_out_params = load_and_fit_gaussian(file_out_path)
        predicted_label = determine_model(fitted_in_params, fitted_out_params)
        results.append({'KL(in||out)': predicted_label})

    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <aug_index>")
        sys.exit(1)

    aug_index = sys.argv[1]
    gaussian_params_path = f'/users/home/parentfolder/Augment/result/gaussian_params_{aug_index}.csv'
    results_dir = './result/'

    # Load Gaussian parameters
    gaussian_params = pd.read_csv(gaussian_params_path).set_index('Model')

    # Process files and get results
    results = process_files(aug_index, results_dir)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/users/home/parentfolder/Augment/attack_result/kl_divergence/label_predictions_KL_{aug_index}.csv', index=False)

    print("Processing complete. Results saved.")
