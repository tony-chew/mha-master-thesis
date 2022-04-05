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
        self.hyp_tag = setup_cfg["setup"]["hyperparameters"]["hyperparameter_tag"]
        self.num_images = setup_cfg["setup"]["num_images"]
        self.print_interval = setup_cfg["setup"]["print_interval"]
        self.num_workers = setup_cfg["setup"]["num_workers"]

        # Test specific setup
        self.test_split = setup_cfg["testing"]["test_split"]
        self.val_split = setup_cfg["testing"]["val_split"]
        self.batch_size = setup_cfg["testing"]["batch_size"]
        self.model_path = setup_cfg["testing"]["model_path"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.val_data_loader, self.test_data_loader = self.dataset()

        # Model, loss and optimiser setup
        self.model_name = model_cfg["model"]
        if self.mode == 'semseg':
            self.test_mode = setup_cfg["testing"]["test_mode"]
            self.num_classes = model_cfg[self.model_name]["num_classes"]
            self.running_metrics = utils.SegmentationMetric(self.num_classes)
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.loss_dict = setup_cfg["setup"]["loss"]
        self.device = self.device_check()

        # Visualisation setup
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]
        if self.mode == 'autoencoder':
            self.visualise_val = utils.Visualise(dataloader=self.val_data_loader,
                                                 num_images=self.num_images,
                                                 device=self.device)
            self.visualise_test = utils.Visualise(dataloader=self.test_data_loader,
                                                  num_images=self.num_images,
                                                  device=self.device)
        elif self.mode == 'semseg':
            self.visualise_semseg_val = utils.VisualiseSemSeg(num_images=self.num_images)

        # Pathfiles setup
        self.models_save_path = pathlib.Path(MODELS) / self.model_name / self.dataset_name
        utils.mkdir(self.models_save_path)
        self.results_save_path = pathlib.Path(RESULTS) / self.model_name / self.dataset_name / "test"
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
                                split=self.test_split,
                                input_res=self.input_res,
                                test_mode=self.test_mode if self.mode == 'semseg' else False)
        test_data_loader = test_data.load_data(num_workers=self.num_workers,
                                               shuffle=False)

        val_data = data_loader(**data_loader_params,
                               augments=None,
                               split=self.val_split,
                               input_res=self.input_res,
                               test_mode=self.test_mode if self.mode == 'semseg' else False)
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

    def loss(self):
        """
        Establish the loss

        :return loss_function: the chosen loss
        """
        loss_name, loss_params = losses.get_loss(self.loss_dict,
                                                 self.device)
        loss_function = loss_name(**loss_params)

        return loss_function

    def device_check(self):
        """
        Get the best device to test the model on

        :return device: the best device
        """
        # device = utils.best_device(utils.devices())
        device = torch.device(0)
        logging.info(f"Using device: {torch.cuda.get_device_name(device)}")

        return device

    def load_checkpoint(self):
        """
        Load in the model and optimiser state

        :return model: the model for which inferencing is performed on
        """
        if pathlib.Path(self.model_path).is_file():
            checkpoint = torch.load(self.model_path, map_location='cuda')

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

    def create_summary(self,
                       model):
        """
        Utilise torchinfo to display the architecture of the model in tabular form

        :param model: the model of interest
        :param model_summary: a summary of the model
        """
        C, H, W = self.test_data_loader.dataset[0][0].shape
        model_summary = torchinfo.summary(model, input_size=(self.batch_size, C, H, W))

        return model_summary

    def test_autoencoder(self,
                         model,
                         loss_function,
                         best_fps):
        """
        Run inferencing on the model for the autoencoder, gather inference results and visualisations and save them locally

        :param model: the model of interest
        :param loss_function: the chosen loss
        :param best_fps: the test set fps comparator
        :return test_loss: the average test loss of the test set
        :return best_fps: the fps metric of the test set
        """
        size = len(self.test_data_loader.dataset)
        test_loss_meter = utils.AverageTracker()
        num_samples = int(len(self.val_data_loader) * 0.25)
        latent_space = utils.VisualiseLatentSpace(model=model,
                                                  mean=torch.tensor(self.mean),
                                                  std=torch.tensor(self.std),
                                                  num_samples=num_samples)
        model.eval()

        with torch.no_grad():
            # test sequence
            for i, (xs, _, _) in enumerate(self.test_data_loader):
                xs = xs.to(self.device)
                start_t = perf_counter()
                ps = model(xs)
                end_t = perf_counter()
                loss = loss_function(ps, xs)

                test_loss_meter.update(loss.item())
                #
                # fps = utils.calc_fps(self.batch_size, end_t, start_t)
                # if fps > best_fps:
                #     best_fps = fps

                if i % self.print_interval == 0:
                    loss, current = loss.item(), i * len(xs)

                    ps_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                                std=torch.tensor(self.std),
                                                                input=xs,
                                                                output=ps,
                                                                mode=self.mode)
                    pixel_loss = loss_function(ps_inv, xs_inv)
                    self.visualise_test.update(xs_inv, ps_inv)

                    # print(f"test loss: {loss}, pixel test loss: {pixel_loss}, fps: {fps}, [{current} / {size}]")
                    print(f"test loss: {loss}, pixel test loss: {pixel_loss}, [{current} / {size}]")

            test_loss = test_loss_meter.avg

            # validation sequence
            for i, (xs, ys, _) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                start_t = perf_counter()
                ps = model(xs)
                end_t = perf_counter()

                ps_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                            std=torch.tensor(self.std),
                                                            input=xs,
                                                            output=ps,
                                                            mode=self.mode)
                self.visualise_val.update(xs_inv, ps_inv)

                latent_space.update(xs)

                fps = utils.calc_fps(self.batch_size, end_t, start_t)
                if fps > best_fps:
                    best_fps = fps

            latent_space.visualise(self.models_save_path)

        return test_loss, best_fps

    def test_semseg(self,
                    model,
                    loss_function,
                    best_fps):
        """
        Run inferencing on the model for the segmentation, gather inference results and visualisations and save them locally

        :param model: the model of interest
        :param loss_function: the chosen loss
        :param best_fps: the test set fps comparator
        :return val_loss: validation loss
        :return best_fps: the fps metric of the val set (cityscapes do not provide test set labels)
        :return pix_acc: pixel accuracy
        :return mIoU: mean itersection over union of the dataset (main metric)
        :return class_iou: the mIoU of each class category
        """
        size = len(self.val_data_loader)
        val_loss_meter = utils.AverageTracker()
        model.eval()

        with torch.no_grad():
            for i, (xs, ys, gt) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                ys = ys.to(self.device)

                start_t = perf_counter()
                ps = model(xs)
                end_t = perf_counter()

                loss = loss_function(ps, ys)
                val_loss_meter.update(loss.item())
                fps = utils.calc_fps(self.batch_size,
                                     end_t,
                                     start_t)
                if fps > best_fps:
                    best_fps = fps

                # Get feature maps
                if i == 0:
                    utils.visualise_segmentation_features(model,
                                                          xs,
                                                          self.models_save_path)

                # Get semseg metrics
                pred = torch.argmax(ps, 1)
                pred = pred.to("cpu").data.numpy()
                self.running_metrics.update(pred, ys.to("cpu").numpy())

                if i % self.print_interval == 0:
                    loss, current = loss.item(), i * len(xs)
                    print(f"val loss: {loss}, best fps: {best_fps}, [{current} / {size}]")

                # Get colour map and visualise
                pred_img = pred.squeeze(0)
                mask = utils.get_colour_pallete(pred_img)
                mask_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(tuple(self.mean)),
                                                              std=torch.tensor(tuple(self.std)),
                                                              input=xs,
                                                              output=mask,
                                                              mode=self.mode)
                self.visualise_semseg_val.update(xs_inv.to("cpu"), mask_inv, gt)

        score, class_iou = self.running_metrics.get()
        mIoU = score["Mean IoU"]
        pix_acc = score["Overall Acc"]
        val_loss = val_loss_meter.avg

        return val_loss, best_fps, pix_acc, mIoU, class_iou

    def infer(self):
        """
        The complete inference process
        - load in the model and loss conditions
        - conduct inferencing
        - save the metrics and model summary
        - output the visualisation results to tensorboard
        """
        best_fps = -1
        writer = utils.Logger(self.results_save_path)

        # Load conditions
        model = self.load_checkpoint()
        logging.info(f"Loaded model: {type(model).__name__}")
        loss_function = self.loss()
        logging.info(f"Using loss: {type(loss_function).__name__}")

        # Begin testing
        if self.mode == 'autoencoder':
            test_loss, best_fps = self.test_autoencoder(model,
                                                        loss_function,
                                                        best_fps)
            test_set_metrics = {
                "test loss": test_loss,
                "best fps": best_fps
            }

        elif self.mode == 'semseg':
            val_loss, best_fps, pix_acc, mean_iou, class_iou = self.test_semseg(model,
                                                                                loss_function,
                                                                                best_fps)
            test_set_metrics = {
                "val loss": val_loss,
                "best fps": best_fps,
                "pixel acc": pix_acc,
                "mean iou": mean_iou,
                "class_iou": class_iou
            }

        params_filename = self.models_save_path / f"best_model_metrics.{HYPER_EXT}"
        utils.save_contents(test_set_metrics,
                            params_filename,
                            self.mode)

        # Output model summary
        model_summary = self.create_summary(model)
        model_summary_filename = self.models_save_path / f"model_summary.{HYPER_EXT}"
        utils.save_contents(model_summary,
                            model_summary_filename,
                            self.mode)

        # Gather visualisation results
        if self.mode == 'autoencoder':
            self.visualise_test.create_visuals(writer,
                                               "test")
            self.visualise_val.create_visuals(writer,
                                              "val")

        elif self.mode == 'semseg':
            self.visualise_semseg_val.create_visuals(writer, self.models_save_path)

        writer.close()


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
