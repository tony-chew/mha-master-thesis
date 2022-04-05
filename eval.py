#!/usr/bin/env python3

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

RESULTS = "runs/"
MODELS = "models/"
HYPER_EXT = "txt"


@dataclasses.dataclass
class ProgramArguments:
    setup_config_path : str
    model_config_path : str


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


class Evaluator:
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
        self.hyp_tag = setup_cfg["setup"]["hyperparameters"]["hyperparameter_tag"]
        self.num_images = setup_cfg["setup"]["num_images"]
        self.print_interval = setup_cfg["eval"]["print_interval"]
        self.num_workers = setup_cfg["setup"]["num_workers"]

        # Test specific setup
        self.test_split = setup_cfg["testing"]["test_split"]
        self.val_split = setup_cfg["testing"]["val_split"]
        self.batch_size = setup_cfg["testing"]["batch_size"]
        self.model_path = setup_cfg["testing"]["model_path"]
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.val_data_loader, self.test_data_loader = self.dataset()
        self.val_size = len(self.val_data_loader)
        self.test_size = len(self.test_data_loader)

        # Model setup
        self.model_name = model_cfg["model"]
        if self.mode == 'semseg' or self.mode == 'mha':
            self.num_classes = model_cfg[self.model_name]["num_classes"]
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.device = self.device_check()

        # Pathfiles setup
        self.eval_save_path = pathlib.Path(setup_cfg["eval"]["save_path"])
        utils.mkdir(self.eval_save_path)
        self.seed = setup_cfg.get("seed", 10)

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
                                input_res=self.input_res,
                                split=self.test_split)
        test_data_loader = test_data.load_data(num_workers=self.num_workers,
                                               shuffle=False)

        val_data = data_loader(**data_loader_params,
                               augments=None,
                               input_res=self.input_res,
                               split=self.val_split)
        val_data_loader = val_data.load_data(num_workers=self.num_workers,
                                             shuffle=False)

        logging.info(f"Using dataset: {self.dataset_name}")

        return val_data_loader, test_data_loader

    def model(self):
        """
        Establish the model

        :return model: model to perform inferencing on
        """
        model_name, model_params = architectures.get_arch(self.model_name,
                                                          self.model_dict,
                                                          self.seed,
                                                          self.hyp_tag)
        model = model_name(**model_params)

        return model

    def device_check(self):
        """
        Get the best device to test the model on

        :return device: the best device
        """
        device = utils.best_device(utils.devices())
        logging.info(f"Using device: {torch.cuda.get_device_name(device)}")

        return device

    def load_checkpoint(self):
        """
        Load in the model and optimiser state

        :return model: the model for which inferencing is performed on
        """
        if pathlib.Path(self.model_path).is_file():
            checkpoint = torch.load(self.model_path, map_location='cuda:0')

            if self.hyp_tag:
                model_params = checkpoint["model_parameters"]
                model_name, _ = architectures.get_arch(self.model_name,
                                                       self.model_dict,
                                                       self.seed,
                                                       self.hyp_tag)
                model = model_name(**model_params)
            else:
                model = self.model()

            model.to(self.device)
            model = utils.parallelise_model(model,
                                            checkpoint["model_state"])
            model.eval()

            return model

        else:
            raise NotImplementedError("Re-check checkpoint path if it exists")

    def test_autoencoder(self,
                         model):
        """
        Run evaluation on the model, save outputs as png images

        Autoencoder
        :param model: the model of interest
        """
        model.eval()
        ae_test = self.eval_save_path / "autoencoder_test_images"
        ae_val = self.eval_save_path / "autoencoder_val_images"
        utils.mkdir(ae_test)
        utils.mkdir(ae_val)

        with torch.no_grad():
            # test sequence
            for i, (xs, _, _) in enumerate(self.test_data_loader):
                xs = xs.to(self.device)
                ps = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating test set: [{current} / {self.test_size}]")

                ps_inv, _ = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=ps,
                                                       mode=self.mode)
                ps_inv = torch.squeeze(ps_inv, dim=0)
                ps_inv = tf.transforms.ToPILImage()(ps_inv)
                ps_inv.save(ae_test / f"autoencoder_test_{i + 1}.png")

            # validation sequence
            for i, (xs, ys, _) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                ps = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating val set: [{current} / {self.val_size}]")

                ps_inv, _ = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=ps,
                                                       mode=self.mode)
                ps_inv = torch.squeeze(ps_inv, dim=0)
                ps_inv = tf.transforms.ToPILImage()(ps_inv)
                ps_inv.save(ae_val / f"autoencoder_val_{i + 1}.png")

    def test_semseg(self,
                    model):
        """
        Run evaluation on the model, save outputs as png images

        Semantic segmentation
        :param model: the model of interest
        """
        model.eval()
        seg_test = self.eval_save_path / "segmentation_test_images"
        seg_val = self.eval_save_path / "segmentation_val_images"
        utils.mkdir(seg_test)
        utils.mkdir(seg_val)

        with torch.no_grad():
            for i, (xs, _, _) in enumerate(self.test_data_loader):
                xs = xs.to(self.device)
                pred = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating test set: [{current} / {self.test_size}]")

                pred = torch.argmax(pred, 1)
                pred = pred.to("cpu").data.numpy()
                pred_img = pred.squeeze(0)
                mask = utils.get_colour_pallete(pred_img)
                mask.save(seg_test / f"seg_{i + 1}.png")

            for i, (xs, _, _) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                pred = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating val set: [{current} / {self.val_size}]")

                # Get colour map and save visualisation
                pred = torch.argmax(pred, 1)
                pred = pred.to("cpu").data.numpy()
                pred_img = pred.squeeze(0)
                mask = utils.get_colour_pallete(pred_img)
                mask.save(seg_val / f"seg_{i + 1}.png")

    def test_mha(self,
                 model):
        """
        Run evaluation on the model for mha, save outputs as png images

        mha
        :param model: the model of interest
        """
        model.eval()
        test_images = self.eval_save_path / "multi-head_test_images"
        val_images = self.eval_save_path / "segmentation_val_images"
        utils.mkdir(test_images)
        utils.mkdir(val_images)

        with torch.no_grad():
            for i, (xs, _, _) in enumerate(self.test_data_loader):
                xs = xs.to(self.device)
                preds = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating test set: [{current} / {self.test_size}]")

                # Save semantic output
                pred_semseg = torch.argmax(preds[1], 1)
                pred_semseg = pred_semseg.to("cpu").data.numpy()
                pred_semseg = pred_semseg.squeeze(0)
                mask = utils.get_colour_pallete(pred_semseg)
                mask.save(test_images / f"seg_test_{i + 1}.png")

                # Save autoencoder output
                ps_inv, _ = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=preds[0],
                                                       mode=self.mode)
                ps_inv = torch.squeeze(ps_inv, dim=0)
                ps_inv = tf.transforms.ToPILImage()(ps_inv)
                ps_inv.save(test_images / f"autoencoder_test_{i + 1}.png")

            for i, (xs, _, _) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                preds = model(xs)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Evaluating val set: [{current} / {self.val_size}]")

                # Save semantic output
                pred_semseg = torch.argmax(preds[1], 1)
                pred_semseg = pred_semseg.to("cpu").data.numpy()
                pred_semseg = pred_semseg.squeeze(0)
                mask = utils.get_colour_pallete(pred_semseg)
                mask.save(val_images / f"seg_val_{i + 1}.png")

                # Save autoencoder output
                ps_inv, _ = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=preds[0],
                                                       mode=self.mode)
                ps_inv = torch.squeeze(ps_inv, dim=0)
                ps_inv = tf.transforms.ToPILImage()(ps_inv)
                ps_inv.save(val_images / f"autoencoder_val_{i + 1}.png")

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
        logging.info(f"Loaded model: {type(model).__name__}")

        # Begin testing
        if self.mode == 'autoencoder':
            self.test_autoencoder(model)

        elif self.mode == 'semseg':
            self.test_semseg(model)

        elif self.mode == 'mha':
            self.test_mha(model)


def main():
    args = parse_program_arguments()

    with open(args.setup_config_path) as fp:
        setup_cfg = yaml.safe_load(fp)

    with open(args.model_config_path) as fp:
        model_cfg = yaml.safe_load(fp)

    Evaluator(setup_cfg,
              model_cfg)


if __name__ == '__main__':
    main()
