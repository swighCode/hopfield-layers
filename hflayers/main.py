# Import general modules used for e.g. plotting.
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch
from image_processing import process_image, tensor_to_image

# Import Hopfield-specific modules.
from hflayers import HopfieldLayer
from hflayers.auxiliary.data import LatchSequenceSet #Needed?
from hflayers.auxiliary.data import BitPatternSet

#from distutils.version import LooseVersion

# Import auxiliary modules.
from typing import Optional, List, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Flatten, Linear, LSTM, Module, Sequential
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Linear, Flatten
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Set plotting style.
sns.set()
#Vad är detta?
sys.path.append(r'./AttentionDeepMIL')
device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

class GrayscaleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_vector = process_image(image_path)  # Funktion från annan fil. Den gör bilderna 64x64, greyscale och tensorer.
        return {"data": image_vector}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

num_images = 1  # Hur många bilder?
num_pixels = 64*64  # Vi kan kanske ha mindre bilder.

image_paths_train = ['train1.jpg', 'train2.jpg', 'train3.jpg']  # List of your image paths
image_paths_eval = ['eval1.jpg', 'eval2.jpg', 'eval3.jpg']
dataset_train = GrayscaleImageDataset(image_paths_train)
dataset_eval = GrayscaleImageDataset(image_paths_eval)


# Dataset ska vara bilderna.Batch size är antal bilder
#Vi behöver kanske inte använda detta.
data_loader_train = DataLoader(dataset=dataset_train, batch_size=3)

# Validation set would be here.
data_loader_eval = DataLoader(dataset=dataset_eval, batch_size=3)


def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader
                ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies = [], [], []
    for data, target in data_loader:
        data, target = data.to(device=device), target[0].to(device=device)

        # Process data by Hopfield-based network.
        loss = network.calculate_objective(data, target)[0]

        # Update network parameters.
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        error, prediction = network.calculate_classification_error(data, target)
        accuracy = (prediction == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        errors.append(error)
        losses.append(loss.detach().item())

    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)


def eval_iter(network: Module,
              data_loader: DataLoader
              ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.

    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, errors, accuracies = [], [], []
        for data, target in data_loader:
            data, target = data.to(device=device), target[0].to(device=device)

            # Process data by Hopfield-based network.
            loss = network.calculate_objective(data, target)[0]

            # Compute performance measures of current model.
            error, prediction = network.calculate_classification_error(data, target)
            accuracy = (prediction == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            errors.append(error)
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)


def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, errors, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        losses[r'train'].append(performance[0])
        errors[r'train'].append(performance[1])
        accuracies[r'train'].append(performance[2])

        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval)
        losses[r'eval'].append(performance[0])
        errors[r'eval'].append(performance[1])
        accuracies[r'eval'].append(performance[2])

    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(errors), pd.DataFrame(accuracies)


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_performance(loss: pd.DataFrame,
                     error: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_file: str
                     ) -> None:
    """
    Plot and save loss and accuracy.

    :param loss: loss to be plotted
    :param error: error to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))

    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Loss')

    error_plot = sns.lineplot(data=error, ax=ax[1])
    error_plot.set(xlabel=r'Epoch', ylabel=r'Error')

    accuracy_plot = sns.lineplot(data=accuracy, ax=ax[2])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Accuracy')

    fig.tight_layout()
    fig.savefig(log_file)
    plt.show(fig)

#################################################################
#Code for Hopfield layers begins here


set_seed()
hopfield_lookup = HopfieldLayer(
    input_size=bit_pattern_set.num_bits,
    quantity=len(bit_samples_unique))

output_projection = Linear(in_features=hopfield_lookup.output_size * bit_pattern_set.num_instances, out_features=1)
network = Sequential(hopfield_lookup, Flatten(start_dim=1), output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)

losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=250)


