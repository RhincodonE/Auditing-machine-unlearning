from argparse import ArgumentParser
from typing import List
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch as ch
from torch.cuda.amp import autocast
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

Section('training', 'Hyperparameters').params(
    batch_size=Param(int, 'Batch size', default=512),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    in_model_save_path=Param(str, 'path to in model', required=True),
    out_model_save_path=Param(str, 'path to out model', required=True),

)

Section('data', 'data related stuff').params(
    augment_dataset=Param(str, 'file to store augmented datasets', required=True),
    in_observations=Param(str, 'Path to save in_set model logits', required=True),
    out_observations=Param(str, 'Path to save out_set model logits', required=True),
    gpu=Param(int, 'GPU to use', required=True),
)

@param('data.augment_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('data.gpu')
def make_dataloader(augment_dataset=None, batch_size=None, num_workers=None, gpu=0):
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(f'cuda:{gpu}')), Squeeze()]
    image_pipeline: List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(ch.device(f'cuda:{gpu}'), non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    loaders['augmented'] = Loader(augment_dataset, batch_size=batch_size, num_workers=num_workers,
                                  order=OrderOption.SEQUENTIAL,
                                  pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders

# Model definition (from KakaoBrain)
class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

@param('data.gpu')
def construct_model(gpu=0):
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda(int(gpu))
    return model

def save_logits_to_csv(logits, labels, file_path):
    """Save the logits and true labels to a CSV file."""
    df = pd.DataFrame(logits)
    df['label'] = labels
    df.to_csv(file_path, index=False)
    print(f"Saved logits and labels to {file_path}")

@param('training.lr_tta')
def evaluate(model, loaders, file_path, lr_tta=False):
    model.eval()
    all_logits_at_label = []
    all_labels = []

    with ch.no_grad():
        for ims, labs in tqdm(loaders['augmented']):
            with autocast():
                out = model(ims)  # Output logits for each class
                if lr_tta:
                    out += model(ims.flip(-1))
                logits = out.cpu().numpy()  # Move logits to CPU
                labels = labs.cpu().numpy()  # Move labels to CPU

                # Get the logits at the label position (f(x)_y)
                logits_at_label = logits[np.arange(len(labels)), labels]

                # Store logits_at_label and labels
                all_logits_at_label.append(logits_at_label)
                all_labels.append(labels)

    # Concatenate all logits and labels
    all_logits_at_label = np.concatenate(all_logits_at_label, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Save to CSV
    save_logits_to_csv(all_logits_at_label, all_labels, file_path)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='CIFAR-10 Augmented Data Testing')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders = make_dataloader()

    # Load the in_set model
    model_in = construct_model()
    print(f"Loading model from {config['training.in_model_save_path']}")
    model_in.load_state_dict(ch.load(config['training.in_model_save_path']))

    # Load the out_set model
    model_out = construct_model()
    print(f"Loading model from {config['training.out_model_save_path']}")
    model_out.load_state_dict(ch.load(config['training.out_model_save_path']))

    # Evaluate both models on the augmented dataset and save the logits
    print("Evaluating in_set model on augmented dataset and saving logits:")
    evaluate(model_in, loaders, config['data.in_observations'])

    print("Evaluating out_set model on augmented dataset and saving logits:")
    evaluate(model_out, loaders, config['data.out_observations'])
