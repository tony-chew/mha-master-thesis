import torchvision.transforms as tf
import torch
from transforms.augments import (RandomCrop, RandomScaleCrop, RandomHorizontallyFlip, Scale, Compose)


def get_transform_keys():
    """
    Setup the dictionary to gather the desired transformation/augmentation from torchvision

    :param input_res: the desired resolution of the input image
    :return transform_keys: a dictionary of various transformation functions
    """
    return {"hflip": RandomHorizontallyFlip,
            "scale": Scale,
            "rcrop": RandomCrop,
            "rscale_crop": RandomScaleCrop}


def get_transforms(augments_dict):
    if augments_dict is None:
        return None

    augments = []
    augment_keys = get_transform_keys()

    for aug_key, aug_param in augments_dict.items():
        augments.append(augment_keys[aug_key](aug_param))

    compose = Compose(augments)
    return compose


def inverse_normalise(tensor,
                      mean,
                      std):
    """
    Produces the un-normalized version of the tensor
    """
    unnormalize = tf.Normalize((-mean / std), (1.0 / std))

    return unnormalize(tensor)


def inver_norm_pair(mean,
                    std,
                    input,
                    output,
                    mode):
    """
    Produce the unnormalised loss between input and output

    :param mean: mean to utilise for de-normalisation
    :param std: standard deviation to utilise for de-normalisation
    :param input: input tensor
    :param output: output tensor
    :param mode: the mode of training
    """
    if mode == 'autoencoder' or mode == 'mha':
        ps_inv = inverse_normalise(tensor=output,
                                   mean=mean,
                                   std=std)
    else:
        ps_inv = output

    xs_inv = inverse_normalise(tensor=input,
                               mean=mean,
                               std=std)

    return ps_inv, xs_inv
