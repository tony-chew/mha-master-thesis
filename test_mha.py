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
import losses.loss_mha as loss_mha
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
        self.num_images = setup_cfg["setup"]["num_images"]
        self.print_interval = setup_cfg["setup"]["print_interval"]
        self.num_workers = setup_cfg["setup"]["num_workers"]
        self.master_rank = setup_cfg["setup"]["master_rank"]

        # Test specific setup
        self.test_split = setup_cfg["testing"]["test_split"]
        self.val_split = setup_cfg["testing"]["val_split"]
        self.batch_size = setup_cfg["testing"]["batch_size"]
        self.model_path = setup_cfg["testing"]["model_path"]
        self.ae_model_path = setup_cfg["testing"]["ae_path"]
        self.semseg_model_path = setup_cfg["testing"]["semseg_path"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.val_data_loader, self.test_data_loader = self.dataset()

        # Model, loss and optimiser setup
        self.model_name = model_cfg["model"]
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.model_ae_name = model_cfg["model_ae"]
        self.model_ae_dict = {**model_cfg[self.model_ae_name], **model_cfg["dataset"]}
        self.model_semseg_name = model_cfg["model_semseg"]
        self.model_semseg_dict = {**model_cfg[self.model_semseg_name], **model_cfg["dataset"]}
        self.ae_loss_dict = setup_cfg["setup"]["autoencoder_loss"]
        self.semseg_loss_dict = setup_cfg["setup"]["segmentation_loss"]
        self.loss_weights = setup_cfg["setup"]["loss_weights"]
        self.device = self.device_check()

        # Semseg head setup
        self.num_classes = model_cfg[self.model_name]["num_classes"]
        self.running_metrics = utils.SegmentationMetric(self.num_classes)

        # Visualisation setup
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]
        self.visualise_val_ae = utils.Visualise(dataloader=self.val_data_loader,
                                                num_images=self.num_images,
                                                device=self.device)
        self.visualise_test_ae = utils.Visualise(dataloader=self.test_data_loader,
                                                 num_images=self.num_images,
                                                 device=self.device)
        self.visualise_val_ss = utils.VisualiseSemSeg(num_images=self.num_images)
        self.visualise_test_ss = utils.VisualiseSemSeg(num_images=self.num_images)

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
                                input_res=self.input_res)
        test_data_loader = test_data.load_data(num_workers=self.num_workers,
                                               shuffle=False)

        val_data = data_loader(**data_loader_params,
                               augments=None,
                               split=self.val_split,
                               input_res=self.input_res)
        val_data_loader = val_data.load_data(num_workers=self.num_workers,
                                             shuffle=False)

        logging.info(f"Using dataset: {self.dataset_name}")

        return val_data_loader, test_data_loader

    def model(self,
              model_name,
              model_dict):
        """
        Establish the model

        :return model: model to perform inferencing on
        """
        model_name, model_params = architectures.get_arch(model_name=model_name,
                                                          model_dict=model_dict,
                                                          seed=self.seed)
        model = model_name(**model_params)

        return model

    def loss(self):
        """
        Establish the loss. The loss of both heads are established here and grouped as a list. The MultiHeadLoss is
        then instantiated with this loss list

        :return loss: the multi-head loss function
        """
        # Initialise individual head losses
        ae_loss_name, ae_loss_params = losses.get_loss(self.ae_loss_dict,
                                                       self.master_rank)
        semseg_loss_name, semseg_loss_params = losses.get_loss(self.semseg_loss_dict,
                                                               self.master_rank)

        ae_loss = ae_loss_name(**ae_loss_params).to(self.master_rank)
        semseg_loss = semseg_loss_name(**semseg_loss_params).to(self.master_rank)

        # Compile losses to list and initialise multi-head loss
        loss_all = [ae_loss, semseg_loss]
        loss = losses.loss_mha.MultiHeadLoss(losses=loss_all,
                                             loss_weights=self.loss_weights)

        return loss, (ae_loss_params, semseg_loss_params)

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
        Load in the model and optimiser state. Also load in the single head models for fps comparison

        :return model: the model for which inferencing is performed on
        """
        if pathlib.Path(self.model_path).is_file():
            # mha model
            checkpoint = torch.load(f=self.model_path,
                                    map_location='cpu')
            model = self.model(model_name=self.model_name,
                               model_dict=self.model_dict)
            model.to(self.device)
            model = utils.parallelise_model(model,
                                            checkpoint["model_state"])
            model.eval()

            # single head ae model
            checkpoint_ae = torch.load(f=self.ae_model_path,
                                       map_location='cpu')
            model_ae = self.model(model_name=self.model_ae_name,
                                  model_dict=self.model_ae_dict)
            model_ae.to(self.device)
            model_ae = utils.parallelise_model(model_ae,
                                               checkpoint_ae["model_state"])
            model_ae.eval()

            # single head semseg model
            checkpoint_semseg = torch.load(f=self.semseg_model_path,
                                           map_location='cpu')
            model_semseg = self.model(model_name=self.model_semseg_name,
                                      model_dict=self.model_semseg_dict)
            model_semseg.to(self.device)
            model_semseg = utils.parallelise_model(model_semseg,
                                                   checkpoint_semseg["model_state"])
            model_semseg.eval()

            return model, model_ae, model_semseg

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

    def validation_set(self,
                       model,
                       model_ae,
                       model_semseg,
                       loss_function,
                       best_mha_fps):
        """
        Run inferencing on the model, gather inference results and visualisations
        Also gather latent space and feature space visualisations
        On the validation set -> semantic segmentation and autoencoder results and metrics

        :param model: mha model
        :param model_ae: single head autoencoder model
        :param model_semseg: single head semantic segmentation model
        :param loss_function: the multi-loss function
        :param best_mha_fps: current best mha fps
        :param best_sha_fps: current best sha fps
        """
        val_size = len(self.val_data_loader)
        val_loss_meter = utils.AverageTracker()
        ae_val_loss_meter = utils.AverageTracker()
        ss_val_loss_meter = utils.AverageTracker()

        vis_samples = int(len(self.val_data_loader) * 0.25)
        latent_space = utils.VisualiseLatentSpace(model=model,
                                                  mean=torch.tensor(self.mean),
                                                  std=torch.tensor(self.std),
                                                  num_samples=vis_samples)
        model.eval()

        with torch.no_grad():
            for i, (xs, ys, gt) in enumerate(self.val_data_loader):
                xs = xs.to(self.device)
                ys = ys.to(self.device)

                # mha prediction
                mha_start_t = perf_counter()
                preds = model(xs)
                mha_end_t = perf_counter()

                targets = [xs, ys]
                total_loss, head_losses = loss_function(preds,
                                                        targets)

                val_loss_meter.update(total_loss.item())
                ae_val_loss_meter.update(head_losses[0])
                ss_val_loss_meter.update(head_losses[1])
                mha_fps = utils.calc_fps(self.batch_size,
                                         mha_end_t,
                                         mha_start_t)

                if mha_fps > best_mha_fps:
                    best_mha_fps = mha_fps

                    # sha prediction (for fps evaluation only)
                    sha_start_t = perf_counter()
                    ae_preds = model_ae(xs)
                    semseg_preds = model_semseg(xs)
                    sha_end_t = perf_counter()

                    best_sha_fps = utils.calc_fps(self.batch_size,
                                                  sha_end_t,
                                                  sha_start_t)

                # get semseg metrics
                semseg_pred = torch.argmax(preds[1], 1)
                semseg_pred = semseg_pred.to("cpu").data.numpy()
                self.running_metrics.update(semseg_pred, ys.to("cpu").numpy())

                if i % self.print_interval == 0:
                    val_loss, current = total_loss.item(), i * len(xs)
                    print(f"val loss: {val_loss}, best fps: {best_mha_fps}, [{current} / {val_size}]")

                # visualise semseg
                pred_img = semseg_pred.squeeze(0)
                mask = utils.get_colour_pallete(pred_img)
                mask_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(tuple(self.mean)),
                                                              std=torch.tensor(tuple(self.std)),
                                                              input=xs,
                                                              output=mask,
                                                              mode="semseg")
                self.visualise_val_ss.update(xs_inv.to("cpu"),
                                             mask_inv,
                                             gt)

                # viualise segmentation feature maps
                if i == 0:
                    utils.visualise_segmentation_features(model,
                                                          xs,
                                                          self.models_save_path)

                # visualise autoencoder
                ps_inv, _ = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=preds[0],
                                                       mode=self.mode)
                self.visualise_val_ae.update(xs_inv,
                                             ps_inv)

                # visualise autoencoder latent space
                latent_space.update(xs)
            latent_space.visualise(self.models_save_path)

        # semseg metrics
        score, class_iou = self.running_metrics.get()
        mIoU = score["Mean IoU"]
        pix_acc = score["Overall Acc"]

        # loss metrics
        val_loss = val_loss_meter.avg
        ae_val_loss = ae_val_loss_meter.avg
        ss_val_loss = ss_val_loss_meter.avg

        return val_loss, ae_val_loss, ss_val_loss, best_mha_fps, best_sha_fps, pix_acc, mIoU, class_iou

    def test_set(self,
                 model,
                 model_ae,
                 model_semseg,
                 best_mha_fps):
        """
        Run inferencing on the model, gather inference visualisations
        On the test set -> semantic segmentation and autoencoder results and metrics (no loss recorded as cityscapes
        do not provide the test labels)

        :param model: model
        :param best_fps: current best fps
        """
        test_size = len(self.test_data_loader)
        model.eval()

        with torch.no_grad():
            for i, (xs, _, _) in enumerate(self.test_data_loader):
                xs = xs.to(self.device)

                start_t = perf_counter()
                preds = model(xs)
                end_t = perf_counter()

                fps = utils.calc_fps(self.batch_size,
                                     end_t,
                                     start_t)

                if fps > best_mha_fps:
                    best_mha_fps = fps

                    # sha prediction (for fps evaluation only)
                    sha_start_t = perf_counter()
                    ae_preds = model_ae(xs)
                    semseg_preds = model_semseg(xs)
                    sha_end_t = perf_counter()

                    best_sha_fps = utils.calc_fps(self.batch_size,
                                                  sha_end_t,
                                                  sha_start_t)

                if i % self.print_interval == 0:
                    current = i * len(xs)
                    print(f"Test Set, best fps: {best_mha_fps}, [{current} / {test_size}]")

                # visualise autoencoder
                ps_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(self.mean),
                                                       std=torch.tensor(self.std),
                                                       input=xs,
                                                       output=preds[0],
                                                       mode=self.mode)
                self.visualise_test_ae.update(xs_inv,
                                              ps_inv)

        return best_mha_fps, best_sha_fps

    def infer(self):
        """
        The complete inference process
        - load in the model and loss conditions
        - conduct inferencing
        - save the metrics and model summary
        - output the visualisation results to tensorboard
        """
        best_mha_fps_start = -1
        writer = utils.Logger(self.results_save_path)

        # Load conditions
        model, model_ae, model_semseg = self.load_checkpoint()
        logging.info(f"Loaded model: {type(model).__name__}")
        loss_function, loss_params = self.loss()
        ae_loss_params, ss_loss_params = loss_params[0], loss_params[1]
        logging.info(f"Using loss: {type(loss_function).__name__}")

        # Begin testing
        val_results = self.validation_set(model=model,
                                          model_ae=model_ae,
                                          model_semseg=model_semseg,
                                          loss_function=loss_function,
                                          best_mha_fps=best_mha_fps_start)
        val_loss, ae_val_loss, ss_val_loss, best_mha_fps, best_sha_fps, pix_acc, mIoU, class_iou = val_results

        best_mha_fps_test, best_sha_fps_test = self.test_set(model=model,
                                                             model_ae=model_ae,
                                                             model_semseg=model_semseg,
                                                             best_mha_fps=best_mha_fps_start)

        test_set_metrics = {
            "total_val_loss": val_loss,
            "ae_val_loss": ae_val_loss,
            "ss_val_loss": ss_val_loss,
            "best_mha_fps_val": best_mha_fps,
            "best_sha_fps_val": best_sha_fps,
            "best_mha_fps_test": best_mha_fps_test,
            "best_sha_fps_test": best_sha_fps_test,
            "pixel acc": pix_acc,
            "mean iou": mIoU,
            "class_iou": class_iou
        }

        # Save metrics
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
        self.visualise_val_ae.create_visuals(writer,
                                             "val")
        self.visualise_test_ae.create_visuals(writer,
                                              "test")

        self.visualise_val_ss.create_visuals(writer,
                                             self.models_save_path)
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
