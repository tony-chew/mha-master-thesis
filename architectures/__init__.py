import numpy as np
import architectures
import utils
from architectures.autoencoders.conv_ae import *
from architectures.autoencoders.fastscnn_ae import *
from architectures.autoencoders.lednet_ae import *
from architectures.autoencoders.dabnet_ae import *
from architectures.autoencoders.edanet_ae import *
from architectures.autoencoders.erfnet_ae import *
from architectures.semantic_segmentation.erfnet import *
from architectures.semantic_segmentation.edanet import *
from architectures.mha import multihead_net_1
from architectures.mha import multihead_net_2
from architectures.mha import multihead_net_3
from architectures.mha import multihead_net_4

arch_keys = {"conv_ae": architectures.autoencoders.conv_ae.ConvAE,
             "fastscnn_ae": architectures.autoencoders.fastscnn_ae.FastSCNNAE,
             "lednet_ae": architectures.autoencoders.lednet_ae.LEDNetAE,
             "dabnet_ae": architectures.autoencoders.dabnet_ae.DABNet_AE,
             "edanet_ae": architectures.autoencoders.edanet_ae.EDANet_AE,
             "erfnet_ae": architectures.autoencoders.erfnet_ae.ERFNetAE,
             "erfnet": architectures.semantic_segmentation.erfnet.ERFNet,
             "edanet": architectures.semantic_segmentation.edanet.EDANet,
             "mha1": architectures.mha.multihead_net_1.MultiHeadArchitecture,
             "mha2": architectures.mha.multihead_net_2.MultiHeadArchitecture,
             "mha3": architectures.mha.multihead_net_3.MultiHeadArchitecture,
             "mha4": architectures.mha.multihead_net_4.MultiHeadArchitecture}


def get_arch(model_name,
             model_dict,
             seed,
             hyp_tag=False):
    """
    Prepare the model for training

    :param model_name: chosen name of the model to set up
    :param model_dict: a dictionary of the model parameters
    :param seed: seed for reproducibility
    :param hyp_tag: if hyperparameter search is on or off
    :return arch_name: chosen model
    :return arch_params: the parameters of the model
    """
    if seed is not None:
        np.random.seed(seed)

    if model_dict is None:
        raise NotImplementedError("Model not listed")

    else:
        if model_name not in arch_keys:
            raise NotImplementedError(f"Model {model_name} does not exist")

        if hyp_tag:
            arch_params = {k: np.random.choice(v) if isinstance(v, list) else v for k, v in model_dict.items() if \
                           k != "name" and k != "input_res"}

        elif not hyp_tag:
            arch_params = {k: v for k, v in model_dict.items() if k != "name"}

        return arch_keys[model_name], arch_params
