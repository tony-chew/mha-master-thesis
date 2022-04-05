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
        self.hyp_tag = setup_cfg["setup"]["hyperparameters"]["hyperparameter_tag"]
        self.hyp_tag_runs = setup_cfg["setup"]["hyperparameters"]["num_hyperparam_runs"]
        self.num_images = setup_cfg["setup"]["num_images"]
        self.print_interval = setup_cfg["setup"]["print_interval"]
        self.print_interval_val = setup_cfg["setup"]["print_interval_val"]
        self.num_workers = setup_cfg["setup"]["num_workers"]
        self.multiple_gpu = setup_cfg["setup"]["multi_gpus"]["usage"]

        # Train specific setup
        self.save_interval = setup_cfg["training"]["save_interval"]
        self.val_split = setup_cfg["training"]["val_split"]
        self.epochs = setup_cfg["training"]["epochs"]
        self.batch_size = setup_cfg["training"]["batch_size"]
        self.checkpoint_path = setup_cfg["training"]["checkpoint_path"]
        self.patience = setup_cfg["training"]["patience"]

        # Dataset specific setup
        self.dataset_name = model_cfg["dataset"]["name"]
        self.dataset_dict = model_cfg[self.dataset_name]
        self.input_res = model_cfg["dataset"]["input_res"]
        self.augments = transforms.get_transforms(setup_cfg.get("augments", None))
        self.mean = setup_cfg["setup"]["mean"]
        self.std = setup_cfg["setup"]["std"]

        # Model, loss and optimiser setup
        self.model_name = model_cfg["model"]
        if self.mode == 'semseg':
            self.num_classes = model_cfg[self.model_name]["num_classes"]
            self.running_metrics = utils.SegmentationMetric(self.num_classes)
        self.model_dict = {**model_cfg[self.model_name], **model_cfg["dataset"]}
        self.loss_dict = setup_cfg["setup"]["loss"]
        self.opt = setup_cfg["setup"]["optimiser"]

        # Pathfiles and seed setup
        self.models_save_path = pathlib.Path(MODELS) / self.model_name / self.dataset_name
        self.results_save_path = pathlib.Path(RESULTS) / self.model_name / self.dataset_name / "train"
        utils.mkdir(self.models_save_path)
        self.seed_main = setup_cfg.get("seed", 10)

        # Multiple GPU training setup
        if self.multiple_gpu:
            self.no_of_nodes = setup_cfg["setup"]["multi_gpus"]["nodes"]
            self.gpus_per_node = torch.cuda.device_count()
            self.world_size = self.no_of_nodes * self.gpus_per_node
        self.rank = rank
        self.master_rank = master_rank

        # Begin the training
        self.set_deterministic()
        self.multi_gpu_init()
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

    def multi_gpu_init(self,
                       backend="nccl"):
        """
        Initialises the distribution process if multiple gpus are set

        :param backend: the backend for dist.init_process_group
        """
        if self.multiple_gpu:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '6006'
            dist.init_process_group(backend=backend,
                                    world_size=self.world_size,
                                    rank=self.rank)

            print(f"Rank {self.rank + 1} / {self.world_size} training process initialised")

    def device_check(self,
                     model):
        """
        Move the model to the best device and set the model to run on multiple gpus if set

        :param model: model to be trained on
        :return model: model to be trained on (wrapped around DistributedDataParallel if using multiple gpus)
        """
        model_name = type(model).__name__
        if self.multiple_gpu and torch.cuda.device_count() > 1:
            model.to(self.rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])

        elif not self.multiple_gpu:
            if self.rank is None:
                self.rank = utils.best_device(utils.devices())
            model.to(self.rank)

        elif self.multiple_gpu and torch.cuda.device_count() == 1:
            raise NotImplementedError("Re-check multi-gpu status: only one device exists")

        return model, model_name

    def dataset(self):
        """
        Load the train and validation set of the chosen dataset, if multiple gpus are set also load in
        DistributedSampler

        :return train_data_loader: the dataloader of the train set
        :return val_data_loader: the dataloader of the validation set
        """
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

        if self.multiple_gpu:
            train_sampler = DistributedSampler(dataset=training_data.get_dataset(),
                                               rank=self.rank,
                                               num_replicas=self.world_size)
            val_sampler = DistributedSampler(dataset=val_data.get_dataset(),
                                             rank=self.rank,
                                             num_replicas=self.world_size)

        train_data_loader = training_data.load_data(batch_size=self.batch_size,
                                                    num_workers=self.num_workers,
                                                    shuffle=False if self.multiple_gpu else True,
                                                    sampler=train_sampler if self.multiple_gpu else None)

        val_data_loader = val_data.load_data(batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             sampler=val_sampler if self.multiple_gpu else None)

        return train_data_loader, val_data_loader

    def model(self, seed):
        """
        Establish the model

        :return model: chosen model
        :return model_params: the parameters of the chosen model
        """
        model_name, model_params = architectures.get_arch(self.model_name,
                                                          self.model_dict,
                                                          seed,
                                                          self.hyp_tag)
        model = model_name(**model_params)

        return model, model_params

    def loss(self):
        """
        Establish the loss

        :return loss_function: the chosen loss
        :return loss_params: the parameters of the chosen loss
        """
        loss_name, loss_params = losses.get_loss(self.loss_dict,
                                                 self.master_rank)

        loss_name = loss_name(**loss_params)

        return loss_name, loss_params

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
                                                  seed,
                                                  self.hyp_tag)

        optimiser = opt_name(model.parameters(),
                             **opt_params)

        optimiser_params = optimisers.get_opt_params(optimiser)

        return optimiser, optimiser_params

    def load_checkpoint(self,
                        scheduler):
        """
        If set, load in checkpoint. If a hyperparameter search run is also set, establish the model
        based on the checkpoint model parameters

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
                                                   seed=None,
                                                   hyp_tag=self.hyp_tag)
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
            hyp_run = checkpoint["hyperparameter_run"]
            params = checkpoint["parameters"]
            model_params = checkpoint["model_parameters"]
            hyp_results = checkpoint["hyp_results"]
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

            return hyp_run, hyp_results, epoch_count, model, optimiser, scheduler, \
                   results, params, model_params, stop_tracker, best_val, best_val_epochs, best_mean_iou, best_class_iou

        else:
            raise NotImplementedError("Re-check checkpoint path if it exists")

    def save_best_model(self,
                        best_model,
                        best_hyperparams,
                        best_model_params,
                        best_results):
        """
        Save the best model out of the hyperparameter search

        :param best_model: the best model to be saved
        :param best_hyperparams: the hyperparameters of the best hyperparameter search runs
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
              scheduler):
        """
        The training process for one epoch

        :param model: model to be trained on
        :param train_dataloader: the dataset to train on
        :param loss_function: the chosen loss to perform iteration on
        :param optimiser: the chosen optimiser to perform iteration on
        :param scheduler: the scheduling strategy to adjust the learning rate
        :return train_loss: the average training loss of that epoch
        :return optimiser: the optimiser
        :return scheduler: the scheduler
        """
        size = len(train_dataloader.dataset)
        train_loss_meter = utils.AverageTracker()
        model.train()
        print_count = -1

        for i, (xs, ys, _) in enumerate(train_dataloader):
            xs = xs.to(self.rank)
            ps = model(xs)

            if self.mode == "autoencoder":
                loss = loss_function(ps, xs)
            elif self.mode == "semseg":
                ys = ys.to(self.rank)
                loss = loss_function(ps, ys)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

            train_loss_meter.update(loss.item())
            lr = scheduler.get_lr()

            if i % self.print_interval == 0:
                loss = loss.item()
                current = utils.print_progress(self.multiple_gpu,
                                               self.rank,
                                               len(xs),
                                               print_count,
                                               self.print_interval,
                                               i)

                if self.mode == "autoencoder":
                    ps_inv, xs_inv = transforms.inver_norm_pair(mean=torch.tensor(tuple(self.mean)),
                                                                std=torch.tensor(tuple(self.std)),
                                                                input=xs,
                                                                output=ps,
                                                                mode=self.mode)
                    pixel_loss = loss_function(ps_inv,
                                               xs_inv)
                    print(f"GPU: {self.rank}, training loss: {loss}, pixel training loss: {pixel_loss}, lr: {lr} "
                          f"[{current} / {size}]")

                elif self.mode == "semseg":
                    print(f"GPU: {self.rank}, training loss: {loss}, lr: {lr}, [{current} / {size}]")

                print_count += 1 if print_count != 1 else print_count

        train_loss = utils.reduce_values(train_loss_meter.avg,
                                         self.rank,
                                         self.master_rank) if self.multiple_gpu else train_loss_meter.avg

        return train_loss, optimiser, scheduler

    def validate(self,
                 model,
                 val_dataloader,
                 loss_function):
        """
        The validation process for one epoch

        :param model: model to be trained on
        :param val_dataloader: the dataset to validate on
        :param loss_function: the chosen loss to gauge validation performance
        :return val_loss: the average validation loss of that epoch
        :return best_fps: the overall fps of the validation process of that epoch
        """
        size = len(val_dataloader.dataset)
        val_loss_meter = utils.AverageTracker()
        best_fps = -1
        print_count = -1
        model.eval()

        if self.mode == "semseg":
            self.running_metrics.reset()

        with torch.no_grad():
            for i, (xs, ys, _) in enumerate(val_dataloader):
                xs = xs.to(self.rank)
                start_t = perf_counter()
                ps = model(xs)
                end_t = perf_counter()

                if self.mode == "autoencoder":
                    loss = loss_function(ps, xs)
                    loss = loss.item()
                    pix_acc, mIoU = None, None

                elif self.mode == "semseg":
                    ys = ys.to(self.rank)
                    loss = loss_function(ps, ys)

                    pred = torch.argmax(ps, 1)
                    pred = pred.to("cpu").data.numpy()
                    self.running_metrics.update(pred, ys.to("cpu").numpy())

                    if i % self.print_interval_val == 0:
                        loss = loss.item()
                        current = utils.print_progress(self.multiple_gpu,
                                                       self.rank,
                                                       len(xs),
                                                       print_count,
                                                       self.print_interval_val,
                                                       i)
                        print(f"Validating on: GPU: {self.rank}, val loss: {loss}, [{current} / {size}]")
                        print_count += 1 if print_count != 1 else print_count

                val_loss_meter.update(loss)
                fps = utils.calc_fps(self.batch_size,
                                     end_t,
                                     start_t)
                if fps > best_fps:
                    best_fps = fps

            if self.mode == "semseg":
                score, class_iou = self.running_metrics.get()
                mIoU = score["Mean IoU"]
                pix_acc = score["Overall Acc"]

        val_loss = utils.reduce_values(val_loss_meter.avg,
                                       self.rank,
                                       self.master_rank) if self.multiple_gpu else val_loss_meter.avg
        best_fps = utils.reduce_values(best_fps,
                                       self.rank,
                                       self.master_rank) if self.multiple_gpu else best_fps

        return val_loss, best_fps, pix_acc, mIoU

    def train_epochs(self,
                     epoch_count,
                     results,
                     model_path=None,
                     hyp_run=None,
                     hyp_results=None,
                     best_val=None,
                     best_mean_iou=None,
                     best_class_iou=None):
        """
        The complete training process (accounting for hyperparameter search)
        - establish the model, loss, optimiser, device, parameters
        - set up the filepaths for the results and models to be saved to
        - if checkpointing is set, load in information
        - conduct training for the given number of epochs
        - save each epoch as a checkpoint
        - save results to tensorboard

        :param epoch_count: the epoch to begin training on (default 0)
        :param results: the list for results of each epoch to be saved to
        :param model_path: filepath for model to be saved to
        :param hyp_run: if hyperparameter search is set, the run to begin training on (default None)
        :param hyp_results: dictionary of the results of each hyperparameter run
        :param best_val: the best validation loss of the run
        :param best_mean_iou: the current best mIoU value
        :param best_class_iou: the corresponding class IoUs of the best mIoU model
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
        if self.multiple_gpu:
            dist.barrier()
            dist.broadcast(tensor=seed_params, src=self.master_rank)

        # Setup training parameters
        model, model_params = self.model(seed_params.item())
        model, model_name = self.device_check(model)
        train_dataloader, val_dataloader = self.dataset()
        optimiser, optimiser_params = self.optimiser(model,
                                                     seed_params.item())
        loss_function, loss_params = self.loss()
        params = {**model_params, **loss_params, **optimiser_params}
        scheduler = optimisers.PolyLR(optimiser=optimiser,
                                      max_batchnum=len(train_dataloader) * self.epochs)
        stop_tracker = utils.EarlyStopping(patience=self.patience,
                                           mode=self.mode)
        truth_tensor = torch.tensor([False]).to(self.rank)

        # Override conditions if checkpointing is set
        if self.checkpoint_path is not None:
            checkpoint = self.load_checkpoint(scheduler)
            hyp_run, hyp_results, epoch_count, model, optimiser, scheduler, \
            results, params, model_params, stop_tracker, best_val, best_val_epochs, best_mean_iou, best_class_iou = checkpoint

            if self.rank == self.master_rank:
                print("-" * 50)
                if hyp_run is not None:
                    logging.info(f"Loaded in checkpoint from hyperparameter run "
                                 f"[{hyp_run} / {self.hyp_tag_runs}], epoch [{epoch_count} / {self.epochs}]")
                else:
                    logging.info(f"Loaded in epoch {epoch_count}")
                print("-" * 50)
            self.checkpoint_path = None

            if epoch_count == self.epochs:
                return model, results, params, hyp_run, hyp_results, model_params, \
                       best_val, best_val_epochs, best_mean_iou, best_class_iou

        # Re-parametrise hyperparameter run conditions and set up results pathways and tensorboard writer
        if self.hyp_tag:
            params["hyperparamter run"] = hyp_run
            if self.rank == self.master_rank:
                print("-" * 50)
                logging.info(f"Beginning hyperparameter run [{hyp_run} / {self.hyp_tag_runs}]")
                print("-" * 50)

            # Set up results pathway
            hyperparam_name = f"hyperparam_run_{hyp_run}"
            hyperparam_path = self.models_save_path
            param_path = hyperparam_path / f"{hyperparam_name}.{HYPER_EXT}"
            utils.mkdir(hyperparam_path)
            writer = utils.Logger(self.results_save_path / hyperparam_name)

            # Set up models pathway
            model_path = self.models_save_path / hyperparam_name
            utils.mkdir(model_path)

        else:
            if self.rank == self.master_rank:
                print("-" * 50)
                logging.info(f"Begin training (no hyperparameter search set)")
                print("-" * 50)
            writer = utils.Logger(self.results_save_path)
            param_path = self.models_save_path / f"{self.model_name}.{HYPER_EXT}"

        # State conditions
        if self.rank == self.master_rank:
            logging.info(f"Using model: {model_name}")
            utils.display_model_type(model)
            logging.info(f"Using dataset: {self.dataset_name}")
            utils.display_gpu(self.multiple_gpu, self.rank)
            logging.info(f"Using optimiser: {type(optimiser).__name__}")
            logging.info(f"Using loss: {type(loss_function).__name__}")

        while epoch_count < self.epochs:
            epoch_count += 1
            if self.rank == self.master_rank:
                print("-" * 50)
                if self.hyp_tag:
                    print(f"hyp run [{hyp_run} / {self.hyp_tag_runs}], epoch [{epoch_count} / {self.epochs}]")
                else:
                    print(f"epoch [{epoch_count} / {self.epochs}]")
                print("-" * 50)

            # training and validation
            if self.multiple_gpu:
                train_dataloader.sampler.set_epoch(epoch_count)
                val_dataloader.sampler.set_epoch(epoch_count)

            train_loss, optimiser, scheduler = self.train(model,
                                                          train_dataloader,
                                                          loss_function,
                                                          optimiser,
                                                          scheduler)

            val_loss, best_fps, pix_acc, mean_iou = self.validate(model,
                                                                  val_dataloader,
                                                                  loss_function)

            # Log results and conduct result aggregation and checkpoint saving on master gpu process
            if self.rank == self.master_rank:
                if self.mode == "autoencoder":
                    print(f"train loss: {train_loss}, val loss: {val_loss}, best val fps: {best_fps}")
                    results.append((epoch_count, best_fps, train_loss, val_loss))

                elif self.mode == "semseg":
                    print(f"train loss: {train_loss}, val loss: {val_loss}, best val fps: {best_fps}, mean iou: {mean_iou}")
                    val_loss = val_loss.to('cpu')
                    results.append((epoch_count, best_fps, train_loss, val_loss, mean_iou))

                if self.hyp_tag:
                    hyp_results[hyp_run] = results

                # Conduct early stopping check
                num_bad_epochs, best_val_epochs, best_mean_iou = stop_tracker.step(val_loss,
                                                                                   mean_iou)
                early_stop_results = stop_tracker.track_last_epoch(model_path,
                                                                   model,
                                                                   epoch_count,
                                                                   results,
                                                                   hyp_results,
                                                                   params,
                                                                   param_path,
                                                                   self.rank,
                                                                   truth_tensor)
                truth_tensor, valid_epoch, valid_results, valid_hyp_results, last_valid_model = early_stop_results

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
                                          hyp_run=hyp_run,
                                          model_params=model_params,
                                          hyp_results=hyp_results,
                                          num_bad_epochs=num_bad_epochs,
                                          best_val_epochs=best_val_epochs,
                                          last_valid_epoch=valid_epoch,
                                          last_valid_epoch_results=valid_results,
                                          last_valid_hyp_results=valid_hyp_results,
                                          last_valid_model=last_valid_model,
                                          best_val=best_val,
                                          best_mean_iou=best_mean_iou,
                                          best_class_iou=best_class_iou)
                    print(f"checkpoint epoch {epoch_count} saved")

            # Broadcast to the other gpus the result of the truth tensor from the master process
            if self.multiple_gpu:
                dist.barrier()
                dist.broadcast(tensor=truth_tensor, src=self.master_rank)

            if truth_tensor:
                if self.rank == self.master_rank:
                    logging.info(f"{self.patience} failed epochs in a row reached. Stopping training early")
                break

        if self.rank == self.master_rank:
            # Output performance metrics to tensorboard
            train_test_epoch = utils.plot_training_vs_validation_vs_accuracy(results,
                                                                             self.mode)
            if self.mode == 'semseg':
                iou_epoch = utils.plot_iou_vs_epoch(results,
                                                    self.mode)

            if self.hyp_tag:
                writer.figure_log(f"Training Evaluation Metrics for hyperparameter run {hyp_run}",
                                  train_test_epoch)

            else:
                writer.figure_log("Training Evaluation Metrics",
                                  train_test_epoch)

                if self.mode == 'semseg':
                    writer.figure_log("mIOU vs Epochs",
                                      iou_epoch)

            utils.save_contents(params,
                                param_path,
                                self.mode,
                                results)

        if self.multiple_gpu:
            dist.destroy_process_group
        writer.close()

        return last_valid_model, results, params, hyp_run, hyp_results, model_params, best_val, best_val_epochs,\
               best_mean_iou, best_class_iou

    def fit(self):
        """
        Conduct the training process accounting for whether hyperparameter search is set or not
        """
        # Initialise counters and seeding
        epoch_count = 0
        results = []

        # Check hyperparameter tag
        if self.hyp_tag:
            hyp_run = 1
            hyp_results = {}
            best_val = 1
            best_hyp_run = 1

            while hyp_run <= self.hyp_tag_runs:
                # Begin training with the hyperparameters specific to this run
                package = self.train_epochs(hyp_run=hyp_run,
                                            epoch_count=epoch_count,
                                            results=results,
                                            hyp_results=hyp_results,
                                            best_val=best_val)
                model, results, params, hyp_run, hyp_results, model_params, best_val, best_val_epochs,\
                best_mean_iou, best_class_iou = package

                # Check if current epoch is best model; record and save if so
                if self.rank == self.master_rank:
                    if best_val_epochs < best_val:
                        best_val = best_val_epochs
                        best_model, best_model_params = model, model_params
                        best_hyperparams = params
                        best_results = results
                        best_hyp_run = hyp_run

                        best_hyperparams["best_val"] = best_val
                        self.save_best_model(best_model,
                                             best_hyperparams,
                                             best_model_params,
                                             best_results)
                        logging.info(f"New best hyp run: {best_hyp_run}")

                # Reset counters
                epoch_count, results = utils.reset_counters()
                hyp_run += 1

            if self.rank == self.master_rank:
                # Plot the hyperparameter runs
                hyp_tax, hyp_vax = utils.plot_hyp_runs(hyp_results)
                writer = utils.Logger(self.results_save_path / "hyperparameter_results")

                writer.figure_log(f"Training loss for each hyperparameter run",
                                  hyp_tax)
                writer.figure_log(f"Validation loss for each hyperparameter run",
                                  hyp_vax)
                writer.close()

                print("-" * 50)
                logging.info("Hyperparameter search finished")
                print("-" * 50)

        elif not self.hyp_tag:
            # Begin training (one run)
            package = self.train_epochs(model_path=self.models_save_path,
                                        epoch_count=epoch_count,
                                        results=results)

            model, results, params, _, _, model_params, _, best_val, best_mean_iou, best_class_iou = package

            # Save the results of the model version with the best validation metric
            if self.rank == self.master_rank:
                params["best_val"] = best_val

                if self.mode == 'semseg':
                    params["best_mean_iou"] = best_mean_iou

                self.save_best_model(model,
                                     params,
                                     model_params,
                                     results)

            if self.rank == self.master_rank:
                print("-" * 50)
                logging.info("Training finished")
                print("-" * 50)


def main():
    args = parse_program_arguments()

    with open(args.setup_config_path) as fp:
        setup_cfg = yaml.safe_load(fp)

    with open(args.model_config_path) as fp:
        model_cfg = yaml.safe_load(fp)

    multi_gpus = setup_cfg["setup"]["multi_gpus"]["usage"]
    master_rank = setup_cfg["setup"]["master_rank"]

    if multi_gpus:
        mp.spawn(Trainer,
                 args=(master_rank, setup_cfg, model_cfg),
                 nprocs=torch.cuda.device_count())
    else:
        Trainer(rank=master_rank,
                master_rank=master_rank,
                setup_cfg=setup_cfg,
                model_cfg=model_cfg)


if __name__ == '__main__':
    main()
