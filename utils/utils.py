import torch
import torchvision
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import glob
import logging
import pathlib
import collections
from PIL import Image

CPU, CUDA = "cpu", "cuda"
MODELS_EXT = "pth"


EpochResults = collections.namedtuple(
    "epoch_results",
    "epoch_count best_fps train_loss val_loss")


def devices():
    """
    Collect the available devices.

    :return: collection of available devices
    """
    result = set([CPU])

    if torch.cuda.is_available():
        result.add(CUDA)

    return result


def best_device(available_devices=None):
    """
    Select best/fastest device for learning.

    :param available_devices: devices to select from; if set to none, the
                              available devices are enumerated internally
    :return: best/fastest device for learning
    """
    if available_devices is None:
        available_devices = devices()

    if 0 == len(available_devices):
        raise Exception("no device available")

    # GPU is the best choice we can make
    if CUDA in available_devices:
        return CUDA

    # return one of the available devices
    return next(iter(available_devices))


def recursive_glob(rootdir=".",
                   suffix=""):
    """
    Performs recursive glob with given suffix and rootdir

    :param rootdir: root directory of interest
    :param suffix: suffix to be searched
    :return: overall list of the filepaths
    """
    path = pathlib.Path(rootdir)

    return list(path.rglob(pattern=suffix))


def mkdir(path):
    """
    create pathway if it does not exist
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def calc_fps(batch_size,
             end_time,
             start_time):
    """
    Calculate fps to process one image
    """
    return batch_size / (end_time - start_time)


def image_shape(data_loader,
                num_images=1):
    """
    Output the shape of the image as a list [N, C, H, W], default N = 1
    """
    img_shape = list(data_loader.dataset[0][0].shape)
    img_shape.insert(0, num_images)

    return img_shape


def stack_images(xps):
    """
    Concatenate a tuple of tensors along the N dimension
    """
    stk, stk_others = xps[0], xps[1:]
    for i, xs in enumerate(stk_others):
        stk = torch.cat((stk, stk_others[i]), 0)

    return stk


def display_images(xps, num_images=10, normalize=False):
    """
    Form a grid of images of the given tensor set
    """
    return torchvision.utils.make_grid(xps, nrow=num_images, normalize=normalize)


def recons_error_map(xs,
                     ps):
    """
    Calculate the mean square error of an image (set) with its predictions per pixel. The mse is calculated and
    then stacked to 3 channels and then normalised

    :param xs: image tensor
    :param ps: predicted image tensor
    :return: mean square error of the two tensors, normalised
    """
    mse = torch.sum(torch.square(torch.sub(ps, xs)), dim=1)
    mse = torch.stack((mse, mse, mse), dim=1)

    return norm_zero_one(mse)


def norm_zero_one(x):
    """
    Normalises the values in the tensor to values between 0 and 1
    """
    return (x - x.min()) / (x.max() - x.min())


def save_contents(contents,
                  target_path,
                  mode,
                  epoch_results=None):
    """
    Save the contents to target path

    :param contents: a dictionary of parameters of the training run or the model summary
    :param epoch_results: a list of the results of each epoch of the training run
    :param target_path: pathway for text file to save to
    :param mode: the training mode: if training autoencoder, semantic segmentaiton or the multi-head
    :return: a text file of the model/training parameters followed by the subsequent results of each epoch for that
    training run
    """
    f = open(target_path, "w")
    if isinstance(contents, dict):
        for k in contents.keys():
            f.write(f"{k}: {contents[k]}\n")

    if epoch_results is not None:
        f.write("\n")

        if mode == 'autoencoder':
            f.write("epoch count, best_fps, train_loss, val_loss\n")
            epoch_results = convert_to_named(epoch_results)
            for epoch_result in epoch_results:
                f.write(f"{epoch_result.epoch_count}, {epoch_result.best_fps}, "
                        f"{epoch_result.train_loss}, {epoch_result.val_loss}\n")

        elif mode == 'semseg':
            f.write("epoch count, best_fps, train_loss, val loss, mean_iou\n")
            for epoch_result in epoch_results:
                f.write(f"{epoch_result[0]}, {epoch_result[1]}, {epoch_result[2]}, "
                        f"{epoch_result[3]}, {epoch_result[4]}\n")

        elif mode == 'mha':
            f.write("epoch count, best_fps, train_loss, val_loss, ae_val_loss, ss_val_loss, ae_train_loss, ss_train_loss, mean_iou\n")
            for epoch_result in epoch_results:
                f.write(f"{epoch_result[0]}, {epoch_result[1]}, {epoch_result[2]}, "
                        f"{epoch_result[3]}, {epoch_result[4]}, {epoch_result[5]}, "
                        f"{epoch_result[6]}, {epoch_result[7]}, {epoch_result[8]}\n")

    else:
        f.write(str(contents))

    f.close


def semseg_hstack(images):
    """
    Horizontally stack the list of images. Images should be in PIL format
    """
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]

    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]

    return stacked


def semseg_vstack(images):
    """
    Vertically stack the list of images. Images should be in PIL format
    """
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]

    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]

    return stacked


def convert_to_named(contents):
    """
    Convert a list into a namedtuple
    """
    epoch_results = [EpochResults(*el) for el in contents]

    return epoch_results


def display_gpu(multiple_gpu=False,
                device=0):
    """
    Show which gpu(s) we are using for display purposes
    """
    if not multiple_gpu:
        logging.info(f"Using device: {torch.cuda.get_device_name(device)}")

    elif multiple_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using devices: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")


def display_model_type(model):
    """
    Display the instance type of the model
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_type = type(model).__name__
    else:
        model_type = 'torch.nn.Module'

    logging.info(f"Using model type: {model_type}")


def parallelise_model(model,
                      model_state):
    """
    Wrap the model depending on the state of the model

    :param model: the model
    :param model_state: the state of the model
    :return model: the model wrapped as a model.module if DataParallelisation is utilised
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    return model


def save_parallelised_model(model):
    """
    Convert the model depending on if parallelisation is utilised or not

    :param model: the model
    :return model.state_dict(): depending on data parrallelisation
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()


def compute_mean_std(dataloader):
    """
    Compute the mean and std of the dataset

    :param dataloader: the dataset
    :return mean: mean of the dataset
    :return std: std of the dataset
    """
    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, *_ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def reduce_values(result,
                  rank,
                  master_rank):
    """
    Under DistributedDataParallel, aggregate the specified result across the ranks and store it on the master process.
    Written specifically for one node, two gpus usage

    :param result: the result to be aggregated
    :param rank: the current rank of the result that the gpu lies in
    :param master_rank: where the aggregated result is sent to
    """
    result_tensor = torch.tensor([result]).to(rank)
    dist.reduce(tensor=result_tensor,
                op=dist.ReduceOp.SUM,
                dst=master_rank)

    if rank == 0:
        result_tensor = result_tensor / 2.0

    return result_tensor.item()


def print_progress(multiple_gpus,
                   rank,
                   length,
                   print_count,
                   print_interval,
                   iteration):
    """
    Organise which loss and iteration values to display depending on the rank and if multiple gpus are used

    :param multiple_gpus: the tag for whether multiple gpus are used
    :param rank: the gpu number
    :param length: the length of the batch size
    :param print_count: the count of where the loop is
    :param print_interval: the interval with which to display the progress
    :param iteration: the current iteration of the training
    :return current: the iteration number to display
    """
    if multiple_gpus:
        if iteration == 0:
            current = iteration * length
        elif iteration != 0:
            current = iteration * length + (print_count * print_interval * length)

        if rank == 1 and iteration != 0:
            current = current + print_interval * length

    else:
        current = iteration * length

    return current


def rel_difference(x, y):
    """
    Find the relative change between the two argument values

    :param x: original number
    :param y: change to number
    """
    return round(((y - x) / x) * 100, 2)


def get_lr(optimiser):
    """
    Get the optimiser of the model
    """
    for param_group in optimiser.param_groups:

        return param_group['lr']


def get_activation(name,
                   activation):
    """
    Get a snapshot of the convolutional operation
    """
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook