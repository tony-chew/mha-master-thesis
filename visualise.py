# !/usr/bin/env python3

import argparse
import dataclasses
import torch
import torch.utils.data
import yaml
import logging
import pathlib
import torchinfo
import utils
import loaders
import architectures
import losses
import transforms
import train
import numpy as np
from time import perf_counter
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

RESULTS = "runs/"
MODELS = "models/"
HYPER_EXT = "txt"


@dataclasses.dataclass
class ProgramArguments:
    setup_config_path: str
    model_config_path: str


def parse_program_arguments():
    """
    Parse program arguments and create a program configuration.

    :return: program configuration
    """
    parser = argparse.ArgumentParser(
        description="Parser for the configuration file to begin testing of the Autoencoder testing")

    parser.add_argument(
        "--setup_config_path",
        type=str,
        help="Configuration file to use for inferencing")

    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Configuration file for the model setup")

    args = parser.parse_args()

    return ProgramArguments(
        setup_config_path=args.setup_config_path,
        model_config_path=args.model_config_path)


class Tester:
    """
    Testing and its helper functions
    """

    def __init__(self,
                 setup_cfg,
                 model_cfg):
        """
        Read the contents of the config file

        :param setup_cfg: config file of the testing parameters
        :param model_cfg: config file of the model parameters
        """
        # General setup
        logging.basicConfig(level=logging.INFO)
        self.mode = setup_cfg["setup"]["mode"]
        self.num_workers = setup_cfg["setup"]["num_workers"]

        # Test specific setup
        self.test_split = setup_cfg["testing"]["test_split"]
        self.val_split = setup_cfg["testing"]["val_split"]
        self.model_path = setup_cfg["testing"]["model_path"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.val_data_loader, self.test_data_loader = self.dataset()

        # Model, loss and optimiser setup
        self.model_name = model_cfg["model"]
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.device = self.device_check()
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]

        # Pathfiles setup
        self.models_save_path = pathlib.Path(MODELS) / self.model_name / self.dataset_name
        self.results_save_path = pathlib.Path(RESULTS) / self.model_name / self.dataset_name / "test"
        self.seed = setup_cfg.get("seed", 10)

        self.models_save_path = pathlib.Path(MODELS) / self.model_name / self.dataset_name
        utils.mkdir(self.models_save_path)

        # Begin inference
        self.infer()

    def dataset(self):
        """
        Load the test set of the chosen dataset

        :return test_data_loader: the dataloader of the test set
        """
        data_loader, data_loader_params = loaders.get_loader(dataset_name=self.dataset_name,
                                                             dataset_dict=self.dataset_dict,
                                                             mode=self.mode)
        test_data = data_loader(**data_loader_params,
                                augments=None,
                                split=self.test_split,
                                input_res=self.input_res)
        test_data_loader = test_data.load_data(num_workers=self.num_workers,
                                               shuffle=False)

        val_data = data_loader(**data_loader_params,
                               augments=None,
                               split=self.val_split,
                               input_res=self.input_res)
        val_data_loader = val_data.load_data(num_workers=self.num_workers,
                                             shuffle=False)

        return val_data_loader, test_data_loader

    def device_check(self):
        """
        Get the best device to test the model on

        :return device: the best device
        """
        device = utils.best_device(utils.devices())

        return device

    def model(self):
        """
        Establish the model

        :return model: model to perform inferencing on
        """
        model_name, model_params = architectures.get_arch(self.model_name,
                                                          self.model_dict,
                                                          self.seed)
        model = model_name(**model_params)

        return model

    def load_checkpoint(self):
        """
        Load in the model and optimiser state

        :return model: the model for which inferencing is performed on
        """
        if pathlib.Path(self.model_path).is_file():
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

            model = self.model()
            # model.to(self.device)
            model = utils.parallelise_model(model,
                                            checkpoint["model_state"])
            model.eval()

            return model

        else:
            raise NotImplementedError("Re-check checkpoint path if it exists")

    def visualise_semseg(self,
                         model):
        """
        Run inferencing on the model, gather inference results and visualisations
        On the validation set -> semantic segmentation and autoencoder results and metrics

        :param model: model
        :param loss_function: the multi-loss function
        :param best_fps: current best fps
        """
        model.eval()

        with torch.no_grad():
            for i, (xs, ys, gt) in enumerate(self.val_data_loader):
                if i == 1:
                    break

                activation = {}

                model.encoder.encoder[0][0].conv.conv.register_forward_hook(utils.get_activation('conv1',
                                                                                                 activation))
                model.encoder.encoder[1][5].conv3.rw_conv.conv.register_forward_hook(utils.get_activation('conv2',
                                                                                                          activation))
                model.encoder.encoder[2][4].conv3.rw_conv.conv.register_forward_hook(utils.get_activation('conv3',
                                                                                                          activation))

                output = model(xs)

                output_1 = activation['conv1'].squeeze()
                output_2 = activation['conv2'].squeeze()
                output_3 = activation['conv3'].squeeze()

                fig1 = plt.figure(figsize=(30, 20.1))
                ax1= [fig1.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
                for i, a in enumerate(ax1):
                    a.imshow(output_1[i], 'gray')
                    a.axis("off")
                fig1.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(self.models_save_path / "output_one", bbox_inches='tight', pad_inches=0, transparent=True)

                fig2 = plt.figure(figsize=(15, 10.05))
                ax2 = [fig2.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
                for i, a in enumerate(ax2):
                    a.imshow(output_2[i], 'gray')
                    a.axis("off")
                fig2.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(self.models_save_path / "output_two", bbox_inches='tight', pad_inches=0, transparent=True)

                fig3 = plt.figure(figsize=(15, 10.05))
                ax3 = [fig3.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
                for i, a in enumerate(ax3):
                    a.imshow(output_3[i], 'gray')
                    a.axis("off")
                fig3.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(self.models_save_path / "output_three", bbox_inches='tight', pad_inches=0, transparent=True)

                # plt.show()

    def visualise_autoencoder(self,
                              model):
        model.eval()
        pca = PCA(16)
        data = []
        targets = []
        n_samples = int(len(self.val_data_loader) * 0.25)
        counter = 0
        to_image = tf.ToPILImage()

        with torch.no_grad():
            for i, (xs, ys, _) in enumerate(self.val_data_loader):
                output = model.encoder_bottleneck(xs)
                xs = transforms.inverse_normalise(tensor=xs,
                                                  mean=torch.tensor(self.mean),
                                                  std=torch.tensor(self.std))

                output = output.detach().numpy()
                xs = xs.detach()

                for output, xs in zip(output, xs):
                    data.append(output.reshape(512))
                    targets.append(xs)
                    counter += 1

                if counter >= n_samples:
                    break

            data = np.array(data)
            targets = np.array(targets, dtype=object)

            data = data[:int(len(data) * 0.5)]
            targets = targets[:int(len(targets) * 0.5)]
            print(len(data))

            # data = pca.fit_transform(data)
            # print(data.shape)

            tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)
            tx, ty = tsne[:, 0], tsne[:, 1]
            tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
            ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
            width = 8000
            height = 7000

            full_image = Image.new('RGB', (width, height), (255, 255, 255))
            for img, x, y in zip(targets, tx, ty):
                tile = to_image(img)
                full_image.paste(tile, (int((width - 1200) * (x)), int((height - 500) * (y))), mask=tile.convert('RGBA'))

            plt.figure(figsize=(240, 240))
            plt.imshow(full_image)
            plt.axis("off")
            # plt.show()
            full_image.save(self.models_save_path / "autoencoder_latent_vis.png")


    def infer(self):
        """
        The complete inference process
        - load in the model and loss conditions
        - conduct inferencing
        - save the metrics and model summary
        - output the visualisation results to tensorboard
        """
        # Load conditions
        model = self.load_checkpoint()

        if self.mode == 'semseg':
            self.visualise_semseg(model)
        elif self.mode == 'autoencoder':
            self.visualise_autoencoder(model)


def main():
    args = parse_program_arguments()

    with open(args.setup_config_path) as fp:
        setup_cfg = yaml.safe_load(fp)

    with open(args.model_config_path) as fp:
        model_cfg = yaml.safe_load(fp)

    Tester(setup_cfg,
           model_cfg)


if __name__ == '__main__':
    main()
