from torch.optim import SGD, Adam, RMSprop
import numpy as np
from .scheduler import *

opt_keys = {"sgd": SGD,
            "adam": Adam,
            "rmsprop": RMSprop}


def get_opt(opt,
            seed,
            hyp_tag=False):
    """
    Prepare the optimiser for training

    :param opt: a dictionary of the optimiser name and its parameters
    :param seed: seed for reproducibility
    :param hyp_tag: if hyperparameter search is on or off
    :return opt_name: chosen optimiser module
    :return opt_params: the parameters of the chosen optimiser module
    """
    if seed is not None:
        np.random.seed(seed)

    if opt is None:
        raise NotImplementedError("Optimiser not listed")

    else:
        opt_name = opt["name"]
        if opt_name not in opt_keys:
            raise NotImplementedError(f"Optimiser {opt_name} not implemented")

        if hyp_tag:
            opt_params = {k: sample_log_value(v, seed) if isinstance(v, list) else v for k, v in opt.items() if k != "name"}
        elif not hyp_tag:
            opt_params = {k: v for k, v in opt.items() if k != "name"}

        return opt_keys[opt_name], opt_params


def get_opt_params(optimiser):
    """
    Retrieve the parameters directly from the optimiser
    """
    for param_group in optimiser.param_groups:
        parameters = {var: param_group[var] for var in param_group if var != "params"}

    return parameters


def sample_log_value(v,
                     seed):
    """
    Return a randomly sampled value from the list in log format

    :param v: a list of two values with a min and max value
    :param seed: seed for reproducibility
    """
    np.random.seed(seed)
    if len(v) != 2:
        raise NotImplementedError("List of sample range should have only two values (min and max limit)")
    else:
        return 10 ** np.random.uniform(v[0], v[1])