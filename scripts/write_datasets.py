from argparse import ArgumentParser
import numpy as np
import torchvision.transforms as transforms
import torchvision
import os

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from torch.utils.data import Subset

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torch.utils.data import Dataset

Section('data', 'arguments to give the writer').params(
    sample_index=Param(int, 'The index of the sample to augment', required=True),
    augmentations=Param(int, 'Number of augmentations to generate', required=True),
    in_dataset=Param(str, 'file to store in datasets', required=True),
    out_dataset=Param(str, 'file to store out datasets', required=True),
    test_dataset=Param(str, 'file to store test datasets', required=True),
    augment_dataset=Param(str, 'file to store augment datasets', required=True),
)

class AugmentedDataset(Dataset):
    """Custom dataset to hold the augmented samples"""
    def __init__(self, base_sample, augmentations):
        self.base_sample = base_sample
        self.augmentations = augmentations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
        ])

    def __len__(self):
        return self.augmentations

    def __getitem__(self, idx):
        return self.transform(self.base_sample[0]), self.base_sample[1]

@param('data.sample_index')
@param('data.augmentations')
@param('data.in_dataset')
@param('data.out_dataset')
@param('data.test_dataset')
@param('data.augment_dataset')
def main(sample_index, augmentations, in_dataset, out_dataset, test_dataset, augment_dataset):
    # Set seed for reproducibility
    np.random.seed(2)

    # Load the CIFAR-10 datasets (train and test)
    full_train_dataset = torchvision.datasets.CIFAR10('/users/home/parentfolder/Augment/tmp', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10('/users/home/parentfolder/Augment/tmp', train=False, download=True)

    # Split the dataset into "in" and "out"
    all_indices = np.arange(len(full_train_dataset))
    out_indices = np.delete(all_indices, sample_index)  # Out set excludes the sample at `sample_index`

    in_set = full_train_dataset
    out_set = Subset(full_train_dataset, out_indices)

    # Get the sample at the specified index for augmentation
    base_sample = full_train_dataset[sample_index]

    # Create the save directory if it doesn't exist

    # 1. Save the in-set (all 50,000 samples)
    print(f"Saving in-set (50,000 samples) to .beton format")
    writer = DatasetWriter(in_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(in_set)

    # 2. Save the out-set (49,999 samples)
    print(f"Saving out-set (49,999 samples) to .beton format")
    writer = DatasetWriter(out_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(out_set)

    # 3. Generate 1,000 augmentations for the sample at `sample_index`
    print(f"Generating {augmentations} augmentations for sample index {sample_index} (label: {base_sample[1]})")
    augment_set = AugmentedDataset(base_sample, augmentations)

    # 4. Save the augmented sample dataset
    print(f"Saving 1,000 augmented samples to {augment_dataset} in .beton format")

    writer = DatasetWriter(augment_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(augment_set)

    # 5. Save the test set (all 10,000 samples)
    print(f"Saving test set (10,000 samples) to .beton format")
    writer = DatasetWriter(test_dataset, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(test_set)

    print("Finished saving datasets.")

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='CIFAR-10 Dataset Splitter and Augmentation Generator')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
