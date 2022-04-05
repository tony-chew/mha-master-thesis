#!/usr/bin/env python3

import argparse
import dataclasses
import torch
import torchvision
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import yaml
import os
import logging
import numpy as np
import random
import collections
import pathlib
import architectures
import utils
import loaders
import losses
import losses.loss_mha as loss_mha
import optimisers
import transforms
import sys
from time import perf_counter

RESULTS = "runs/"
MODELS = "models/"
MODELS_EXT = "pth"
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
        description="Parser for the configuration file to begin training of the Autoencoder")

    parser.add_argument(
        "--setup_config_path",
        type=str,
        help="Configuration file to use for training setup")

    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Configuration file for the model setup")

    args = parser.parse_args()

    return ProgramArguments(
        setup_config_path=args.setup_config_path,
        model_config_path=args.model_config_path
    )


class Trainer:
    """
    Training and its helper functions
    """
    def __init__(self,
                 rank,
                 master_rank,
                 setup_cfg,
                 model_cfg):
        """
        Read the contents of the config file

        :param rank: if set, the desired gpu(s) to train on and dummy argument for multiprocessing.spawn
        :param master_rank: the chosen rank to be the master process (in most cases it will be rank 0)
        :param setup_cfg: config file of the training parameters
        :param model_cfg: config file of the model parameters
        """
        # General setup
        logging.basicConfig(level=logging.INFO)
        self.mode = setup_cfg["setup"]["mode"]
        self.num_images = setup_cfg["setup"]["num_images"]
        self.print_interval = setup_cfg["setup"]["print_interval"]
        self.print_interval_val = setup_cfg["setup"]["print_interval_val"]
        self.num_workers = setup_cfg["setup"]["num_workers"]

        # Train specific setup
        self.save_interval = setup_cfg["training"]["save_interval"]
        self.val_split = setup_cfg["training"]["val_split"]
        self.epochs = setup_cfg["training"]["epochs"]
        self.batch_size = setup_cfg["training"]["batch_size"]
        self.checkpoint_path = setup_cfg["training"]["checkpoint_path"]
        self.patience = setup_cfg["training"]["patience"]
        self.train_mode = setup_cfg["training"]["mode"]
        if self.train_mode == "stepbystep":
            self.stepbystep_iterations = setup_cfg["training"]["stepbystep_iterations"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.augments = transforms.get_transforms(setup_cfg.get("augments", None))
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]

        # Model, loss and optimiser setup
        self.model_name = model_cfg["model"]
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.ae_loss_dict = setup_cfg["setup"]["autoencoder_loss"]
        self.semseg_loss_dict = setup_cfg["setup"]["segmentation_loss"]
        self.loss_weights = setup_cfg["setup"]["loss_weights"]
        self.opt = setup_cfg["setup"]["optimiser"]

        # Semseg head setup
        self.num_classes = model_cfg[self.model_name]["num_classes"]
        self.running_metrics = utils.SegmentationMetric(self.num_classes)

        # Pathfiles and seed setup
        self.models_save_path = pathlib.Path(MODELS) / self.model_name / self.dataset_name
        self.results_save_path = pathlib.Path(RESULTS) / self.model_name / self.dataset_name / "train"
        utils.mkdir(self.models_save_path)
        self.seed_main = setup_cfg.get("seed", 10)
        self.model_path = setup_cfg["testing"]["model_path"]

        # GPU training setup
        self.rank = rank
        self.master_rank = master_rank

        # Begin the training
        self.set_deterministic()
        self.fit()

    def set_deterministic(self):
        """
        Set up seeds for reproducibility
        """
        torch.manual_seed(self.seed_main)
        torch.cuda.manual_seed(self.seed_main)
        torch.cuda.manual_seed_all(self.seed_main)
        np.random.seed(self.seed_main)
        random.seed(self.seed_main)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def dataset(self):
        """
        Load the train and validation set of the chosen dataset

        :return train_data_loader: the dataloader of the train set
        :return val_data_loader: the dataloader of the validation set
        """
        # Initialise dataloaders
        data_loader, data_loader_params = loaders.get_loader(dataset_name=self.dataset_name,
                                                             dataset_dict=self.dataset_dict,
                                                             mode=self.mode)

        training_data = data_loader(**data_loader_params,
                                    augments=self.augments,
                                    input_res=self.input_res)
        val_data = data_loader(**data_loader_params,
                               augments=None,
                               input_res=self.input_res,
                               split=self.val_split)

        # Load the dataloaders according to batch size
        train_data_loader = training_data.load_data(batch_size=self.batch_size,
                                                    num_workers=self.num_workers,
                                                    shuffle=True)

        val_data_loader = val_data.load_data(batch_size=self.batch_size,
                                             num_workers=self.num_workers)

        return train_data_loader, val_data_loader

    def model(self, seed):
        """
        Establish the model

        :return model: chosen model
        :return model_params: the parameters of the chosen model
        """
        model_name, model_params = architectures.get_arch(self.model_name,
                                                          self.model_dict,
                                                          seed)
        model = model_name(**model_params)

        return model, model_params

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

    def optimiser(self,
                  model,
                  seed):
        """
        Establish the optimiser

        :param model: model to perform optimisation on
        :return optimiser: the chosen optimiser
        :return optimiser_params: the parameters of the chosen optimiser
        """
        opt_name, opt_params = optimisers.get_opt(self.opt,
                                                  seed)

        optimiser = opt_name(filter(lambda p: p.requires_grad, model.parameters()),
                             **opt_params)

        optimiser_params = optimisers.get_opt_params(optimiser)

        return optimiser, optimiser_params

    def configure_mha(self,
                      model,
                      iteration_count):
        """
        In step by step training mode, freeze the segmentation head if iteration count is even (train autoencoder head)
        In step by step training mode, freeze the autoencoder head if iteration count is odd (train segmentation head)

        :param model: the model to look at
        :param iteration_count: the iteration count
        """
        # Load in previous iteration of step by step model (skip if it's the first iteration)
        if self.train_mode == 'stepbystep' and iteration_count != 1:
            checkpoint = torch.load(self.model_path)
            model = utils.parallelise_model(model,
                                            checkpoint["model_state"])

        # iteration count is even -> freeze segmentation head, train autoencoder
        if self.train_mode == "stepbystep" and iteration_count % 2 == 0:
            print(f"Freezing Segmentation Head, training Autoencoder Head, [{iteration_count} / {self.stepbystep_iterations}")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[0] == "segmentation_head":
                    v.requires_grad = False

        # iteration count is odd -> freeze autoencoder head, train segmentation
        elif self.train_mode == "stepbystep" and iteration_count % 2 != 0:
            print(f"Freezing Autoencoder Head, training Segmentation Head, [{iteration_count} / {self.stepbystep_iterations}")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[0] == 'autoencoder_head':
                    v.requires_grad = False

        return model

    def load_checkpoint(self,
                        scheduler):
        """
        If set, load in checkpoint

        :param scheduler: the scheduler to load checkpoint state to
        :return hyp_run: checkpoint run of the hyperparameter search
        :return hyp_results: dictionary of the results of each hyperparameter run
        :return epoch_count: checkpoint epoch count of the hyperparameter search
        :return model: checkpoint model of the hyperparameter search
        :return optimiser: checkpoint optimiser of the hyperparameter search
        :return results: checkpoint results so far of the hyperparameter search
        :return params: parameters of that hyperparameter run
        :return model_params: model parameters of that hyperparameter run
        :return stop_tracker: tally of number of epochs in a row where validation loss was not higher than best val
        """
        if pathlib.Path(self.checkpoint_path).is_file():
            checkpoint = torch.load(f=self.checkpoint_path,
                                    map_location='cpu')

            model_parameters = checkpoint["model_parameters"]
            model_name, _ = architectures.get_arch(model_name=self.model_name,
                                                   model_dict=self.model_dict,
                                                   seed=None)
            model = model_name(**model_parameters)
            model, _ = self.device_check(model)
            optimiser, optimiser_params = self.optimiser(model=model,
                                                         seed=None)

            model = utils.parallelise_model(model,
                                            checkpoint["model_state"])
            last_valid_model = utils.parallelise_model(model,
                                                       checkpoint["last_valid_model_state"])
            optimiser.load_state_dict(checkpoint["optimiser_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            epoch_count = checkpoint["epoch"]
            results = checkpoint["results_state"]
            params = checkpoint["parameters"]
            model_params = checkpoint["model_parameters"]
            best_val = checkpoint["best_val_hyp"]
            best_val_epochs = checkpoint["best_val_epochs"]
            best_mean_iou = checkpoint["best_mean_iou"]
            best_class_iou = checkpoint["best_class_iou"]
            stop_tracker = utils.EarlyStopping(patience=self.patience,
                                               mode=self.mode,
                                               num_bad_epochs=checkpoint["num_bad_epochs"],
                                               best_val=checkpoint["best_val_epochs"],
                                               last_valid_epoch=checkpoint["last_valid_epoch"],
                                               last_valid_epoch_results=checkpoint["last_valid_epoch_results"],
                                               last_valid_hyp_results=checkpoint["last_valid_hyp_results"],
                                               last_valid_model=last_valid_model)
            random.setstate(checkpoint["rand_state"])
            np.random.set_state(checkpoint["np_rand_state"])
            torch.set_rng_state(checkpoint["torch_rand_state"])
            torch.cuda.set_rng_state = checkpoint["torch.cuda_rand_state"]

            return epoch_count, model, optimiser, scheduler, \
                   results, params, model_params, stop_tracker, best_val, best_val_epochs, best_mean_iou, best_class_iou

        else:
            raise NotImplementedError("Re-check checkpoint path if it exists")

    def save_best_model(self,
                        best_model,
                        best_hyperparams,
                        best_model_params,
                        best_results):
        """
        Save the best model with the best results

        :param best_model: the best model to be saved
        :param best_hyperparams: the (hyper)parameters of the best (hyper)parameter search runs
        :param best_model_params: the parameters of the best model
        :param best_results: the full results of the best hyp run
        """
        model_path = self.models_save_path / f"best_model.{MODELS_EXT}"
        best_hyperparam_path = self.models_save_path / f"best_hyperparameters.{HYPER_EXT}"

        state = {
            "model_state": utils.save_parallelised_model(best_model),
            "parameters": best_hyperparams,
            "model_parameters": best_model_params
        }

        torch.save(state,
                   model_path)
        utils.save_contents(best_hyperparams,
                            best_hyperparam_path,
                            self.mode,
                            best_results)

    def train(self,
              model,
              train_dataloader,
              loss_function,
              optimiser,
              scheduler,
              iteration_count):
        """
        The training process for one epoch

        :param model: model to be trained on
        :param train_dataloader: the dataset to train on
        :param loss_function: the chosen loss to perform iteration on
        :param optimiser: the chosen optimiser to perform iteration on
        :param scheduler: the scheduling strategy to adjust the learning rate
        :param iteration_count: if stepbystep mode, the iteration count
        :return train_loss: the average training loss of that epoch
        :return optimiser: the optimiser
        :return scheduler: the scheduler
        """
        size = len(train_dataloader.dataset)
        total_loss_meter = utils.AverageTracker()
        ae_loss_meter = utils.AverageTracker()
        semseg_loss_meter = utils.AverageTracker()

        model.train()

        for i, (xs, ys, _) in enumerate(train_dataloader):
            xs = xs.to(self.rank)
            ys = ys.to(self.rank)
            preds = model(xs)

            targets = [xs, ys]
            total_loss, head_losses = loss_function(preds,
                                                    targets,
                                                    self.train_mode,
                                                    iteration_count=iteration_count)

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()
            scheduler.step()

            total_loss_meter.update(total_loss.item())
            ae_loss_meter.update(head_losses[0])
            semseg_loss_meter.update(head_losses[1])
            lr = scheduler.get_lr()
            lr = float(lr[0])

            if i % self.print_interval == 0:
                current = i * len(xs)

                print(f"GPU: {self.rank}, ae train loss: {head_losses[0]:.5}., seg train loss: {head_losses[1]:.5}, "
                      f"total train loss: {total_loss.item():.5}, lr: {lr:.5}, [{current} / {size}]")

        train_loss = total_loss_meter.avg
        ae_train_loss = ae_loss_meter.avg
        ss_train_loss = semseg_loss_meter.avg

        return train_loss, ae_train_loss, ss_train_loss, optimiser, scheduler

    def validate(self,
                 model,
                 val_dataloader,
                 loss_function,
                 iteration_count):
        """
        The validation process for one epoch

        :param model: model to be trained on
        :param val_dataloader: the dataset to validate on
        :param loss_function: the chosen loss to gauge validation performance
        :param iteration_count: the iteration count
        :return val_loss: the average validation loss of that epoch
        :return best_fps: the overall fps of the validation process of that epoch
        """
        size = len(val_dataloader.dataset)
        ae_loss_meter = utils.AverageTracker()
        semseg_loss_meter = utils.AverageTracker()
        best_fps = -1

        model.eval()
        self.running_metrics.reset()

        with torch.no_grad():
            for i, (xs, ys, _) in enumerate(val_dataloader):
                xs = xs.to(self.rank)
                ys = ys.to(self.rank)
                start_t = perf_counter()
                preds = model(xs)
                end_t = perf_counter()

                targets = [xs, ys]
                total_loss, head_losses = loss_function(preds,
                                                        targets,
                                                        self.train_mode,
                                                        iteration_count)

                ae_loss_meter.update(head_losses[0])
                semseg_loss_meter.update(head_losses[1])

                semseg_pred = torch.argmax(preds[1], 1)
                semseg_pred = semseg_pred.to("cpu").data.numpy()
                self.running_metrics.update(semseg_pred, ys.to("cpu").numpy())

                if i % self.print_interval_val == 0:
                    current = i * len(xs)

                    print(f"Validating on GPU: {self.rank}, autoencoder val loss: {head_losses[0]:.4}, "
                          f"seg val loss: {head_losses[1]:.4}, total val loss: {total_loss.item():.5}, [{current} / {size}]")

                fps = utils.calc_fps(self.batch_size,
                                     end_t,
                                     start_t)
                if fps > best_fps:
                    best_fps = fps

        score, class_iou = self.running_metrics.get()
        mIoU = score["Mean IoU"]
        pix_acc = score["Overall Acc"]
        ae_val_loss = ae_loss_meter.avg
        ss_val_loss = semseg_loss_meter.avg

        return ae_val_loss, ss_val_loss, best_fps, pix_acc, mIoU

    def train_epochs(self,
                     epoch_count,
                     results,
                     model_path=None,
                     best_val=None,
                     best_mean_iou=None,
                     best_class_iou=None,
                     iteration_count=None):
        """
        The complete training process
        - establish the model, loss, optimiser, device, parameters
        - set up the filepaths for the results and models to be saved to
        - if checkpointing is set, load in information
        - conduct training for the given number of epochs
        - save each epoch as a checkpoint
        - save results to tensorboard

        :param epoch_count: the epoch to begin training on (default 0)
        :param results: the list for results of each epoch to be saved to
        :param model_path: filepath for model to be saved to
        :param best_val: the best validation loss of the run
        :param best_mean_iou: the current best mIoU value
        :param best_class_iou: the corresponding class IoUs of the best mIoU model
        :param iteration_count: the iteration count
        :return model: the model that was trained on
        :return results: the results of the whole training process
        :return params: the (hyper)parameters of that training process
        :return hyp_run: the hyperparamter run if set
        :return hyp_results: the results of each hyperparameter run as a dictionary
        :return model_params: the model parameters
        :return writer: the tensorboard writer
        """
        # Setup seeds for parameter initialisation if hyp_tag
        seed_params = torch.tensor([np.random.randint(0, 1000)]).to(self.rank)

        # Setup training parameters
        model, model_params = self.model(seed_params.item())
        model = self.configure_mha(model,
                                   iteration_count)
        model.to(self.rank)
        model_name = type(model).__name__
        train_dataloader, val_dataloader = self.dataset()
        optimiser, optimiser_params = self.optimiser(model,
                                                     seed_params.item())
        loss_function, loss_params = self.loss()
        ae_loss_params, ss_loss_params = loss_params[0], loss_params[1]
        params = {**model_params, **ae_loss_params, **ss_loss_params, **optimiser_params}
        scheduler = optimisers.PolyLR(optimiser=optimiser,
                                      max_batchnum=len(train_dataloader) * self.epochs)
        stop_tracker = utils.EarlyStopping(patience=self.patience,
                                           mode=self.mode)

        # Override conditions if checkpointing is set
        if self.checkpoint_path is not None:
            checkpoint = self.load_checkpoint(scheduler)
            epoch_count, model, optimiser, scheduler, results, params, model_params, stop_tracker, \
            best_val, best_val_epochs, best_mean_iou, best_class_iou = checkpoint

            print("-" * 50)
            logging.info(f"Loaded in epoch {epoch_count}")
            print("-" * 50)
            self.checkpoint_path = None

            if epoch_count == self.epochs:
                return model, results, params, model_params, \
                       best_val, best_val_epochs, best_mean_iou, best_class_iou

        # Set up results pathways and tensorboard writer
        print("-" * 50)
        logging.info(f"Begin training")
        print("-" * 50)

        writer = utils.Logger(self.results_save_path)

        # State conditions
        logging.info(f"Using model: {model_name}")
        utils.display_model_type(model)
        logging.info(f"Using dataset: {self.dataset_name}")
        utils.display_gpu(device=self.rank)
        logging.info(f"Using optimiser: {type(optimiser).__name__}")
        logging.info(f"Using loss: {type(loss_function).__name__}")
        if self.train_mode == 'endtoend':
            param_path = self.models_save_path / f"{self.model_name}.{HYPER_EXT}"
        elif self.train_mode == 'stepbystep':
            param_path = self.models_save_path / f"{self.model_name}_{iteration_count}.{HYPER_EXT}"

        while epoch_count < self.epochs:
            epoch_count += 1
            print("-" * 50)
            print(f"epoch [{epoch_count} / {self.epochs}]")
            print("-" * 50)

            if self.train_mode == "stepbystep" and iteration_count % 2 == 0:
                print(f"Autoencoder only, [{iteration_count} / {self.stepbystep_iterations}]")
            elif self.train_mode == "stepbystep" and iteration_count % 2 != 0:
                print(f"Segmentation only, [{iteration_count} / {self.stepbystep_iterations}]")
            elif self.train_mode == "endtoend":
                print(f"End-to-End mode initialised: Training all heads")

            # training and validation
            train_loss, ae_train_loss, ss_train_loss, optimiser, scheduler = self.train(model,
                                                                                        train_dataloader,
                                                                                        loss_function,
                                                                                        optimiser,
                                                                                        scheduler,
                                                                                        iteration_count)

            ae_val_loss, ss_val_loss, best_fps, pix_acc, mean_iou = self.validate(model,
                                                                                  val_dataloader,
                                                                                  loss_function,
                                                                                  iteration_count)

            # Log results and conduct result aggregation and checkpoint saving on master gpu process
            val_loss = ae_val_loss + ss_val_loss
            print("+" * 100)
            print(f"total train loss: {train_loss:.5}, total val loss: {val_loss:.5}, "
                  f"segmentation mIoU: {mean_iou:.5}, best val fps: {best_fps:.5}")
            print(f"autoencoder val loss: {ae_val_loss:.5}, segmentation val loss: {ss_val_loss:.5}")
            print("+" * 100)
            results.append((epoch_count, best_fps, train_loss, val_loss, ae_val_loss, ss_val_loss, ae_train_loss, ss_train_loss, mean_iou))

            # Conduct early stopping check
            num_bad_epochs, best_val_epochs, best_mean_iou = stop_tracker.step(val_loss,
                                                                               mean_iou)
            early_stop_results = stop_tracker.track_last_epoch(model_path,
                                                               model,
                                                               epoch_count,
                                                               results,
                                                               params,
                                                               param_path,
                                                               self.rank)
            _, valid_epoch, valid_results, _, last_valid_model = early_stop_results

            # Save checkpoint
            if (epoch_count % self.save_interval == 0) and (num_bad_epochs < self.patience) or \
                    (epoch_count == self.epochs):
                utils.save_checkpoint(target_path=model_path,
                                      epoch_count=epoch_count,
                                      model=model,
                                      optimiser=optimiser,
                                      scheduler=scheduler,
                                      results=results,
                                      params=params,
                                      model_params=model_params,
                                      num_bad_epochs=num_bad_epochs,
                                      best_val_epochs=best_val_epochs,
                                      last_valid_epoch=valid_epoch,
                                      last_valid_epoch_results=valid_results,
                                      last_valid_model=last_valid_model,
                                      best_val=best_val,
                                      best_mean_iou=best_mean_iou,
                                      best_class_iou=best_class_iou)
                print(f"checkpoint epoch {epoch_count} saved")

        # Output performance metrics to tensorboard
        train_test_epoch, ae_curve, ss_curve = utils.plot_training_vs_validation_vs_accuracy(results,
                                                                                             self.mode)
        iou_epoch = utils.plot_iou_vs_epoch(results,
                                            self.mode)
        if self.train_mode == 'endtoend':
            writer.figure_log(f"Training Evaluation Metrics",
                              train_test_epoch)
            writer.figure_log(f"Autoencoder Evaluation Metrics",
                              ae_curve)
            writer.figure_log(f"Segmentation Evaluation Metrics",
                              ss_curve)
            writer.figure_log("mIOU vs Epochs",
                              iou_epoch)

        elif self.train_mode == 'stepbystep':
            if iteration_count % 2 == 0:
                status = f"Autoencoder only, [{iteration_count} / {self.stepbystep_iterations}]"
            elif iteration_count % 2 != 0:
                status = f"Segmentation only, [{iteration_count} / {self.stepbystep_iterations}]"

            writer.figure_log(f"Training Evaluation Metrics: {status}",
                              train_test_epoch)
            writer.figure_log(f"Autoencoder Evaluation Metrics: {status}",
                              ae_curve)
            writer.figure_log(f"Segmentation Evaluation Metrics : {status}",
                              ss_curve)
            writer.figure_log(f"mIOU vs Epochs: {status}",
                              iou_epoch)

        utils.save_contents(params,
                            param_path,
                            self.mode,
                            results)
        writer.close()

        return last_valid_model, results, params, model_params, best_val, best_val_epochs, best_mean_iou, best_class_iou

    def fit(self):
        """
        Conduct the training process accounting for whether stepbystep or endtoend training mode is set
        """
        # Initialise counters and seeding
        epoch_count = 0
        results = []

        if self.train_mode == "endtoend":
            # Begin training (one run)
            package = self.train_epochs(model_path=self.models_save_path,
                                        epoch_count=epoch_count,
                                        results=results)

            model, results, params, model_params, _, best_val, best_mean_iou, best_class_iou = package

        elif self.train_mode == "stepbystep":
            iteration_count = 0

            # Begin step by step training according to whether iteration count is odd or even
            while iteration_count < self.stepbystep_iterations:
                iteration_count += 1

                package = self.train_epochs(model_path=self.models_save_path,
                                            epoch_count=epoch_count,
                                            results=results,
                                            iteration_count=iteration_count)
                model, results, params, model_params, _, best_val, best_mean_iou, best_class_iou = package

        # Save the results of the model version with the best validation metric (in stepbystep mode, the final results)
        # ie the last iteration
        params["best_val"] = best_val
        params["best_mean_iou"] = best_mean_iou

        self.save_best_model(model,
                             params,
                             model_params,
                             results)

        print("-" * 50)
        logging.info("Training finished")
        print("-" * 50)


def main():
    args = parse_program_arguments()

    with open(args.setup_config_path) as fp:
        setup_cfg = yaml.safe_load(fp)

    with open(args.model_config_path) as fp:
        model_cfg = yaml.safe_load(fp)

    master_rank = setup_cfg["setup"]["master_rank"]

    Trainer(rank=master_rank,
            master_rank=master_rank,
            setup_cfg=setup_cfg,
            model_cfg=model_cfg)


if __name__ == '__main__':
    main()
