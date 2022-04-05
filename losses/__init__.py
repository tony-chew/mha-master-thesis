from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
import torch
import collections

loss_keys = {"mse": MSELoss,
             "cross entropy": CrossEntropyLoss,
             "L1Loss": L1Loss}


def get_loss(loss,
             rank):
    """
    Prepare the loss for training

    :param loss: a dictionary of the loss name and its parameters
    :param rank: which gpu to set to
    :return loss_name: chosen loss function
    :return loss_params: the parameters of the chosen loss module
    """
    if loss is None:
        raise NotImplementedError("Loss not listed")

    else:
        loss_name = loss["name"]

        if 'weight' in loss.keys():
            cls_weights = loss["weight"]
            cls_weights = torch.tensor(cls_weights)
            cls_weights = cls_weights.to(rank)
            loss["weight"] = cls_weights

        if loss_name not in loss_keys:
            raise NotImplementedError(f"Optimiser {loss_name} not implemented")

        loss_params = {k: v for k, v in loss.items() if k != "name"}

        return loss_keys[loss_name], loss_params
