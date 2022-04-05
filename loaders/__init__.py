import loaders
from .cityscapes import *
from .mapillary import *
from .cifar import *

dataset_keys = {"cityscapes": loaders.CityscapesLoader,
                "mapillary": loaders.MapillaryLoader,
                "cifar100": loaders.CifarLoader}


def get_loader(dataset_name,
               dataset_dict,
               mode):
    """
    Prepare the loader for training or testing

    :param dataset_name: name of dataset to be implemented
    :param dataset_dict: dictionary of parameters for the dataloader
    :param mode: if in semseg or autoencoder mode
    :param encoder_only: whether to train with the encoder or full network
    :return dataset_name: chosen dataset loader
    :return dataset_params: the parameters of the chosen dataset loader
    """
    if dataset_name is None:
        raise NotImplementedError("Dataset not listed")

    else:
        if dataset_name not in dataset_keys:
            raise NotImplementedError(f"dataset {dataset_name} not implemented")

        dataset_params = {k: v for k, v in dataset_dict.items()}
        dataset_params['mode'] = mode

        return dataset_keys[dataset_name], dataset_params
