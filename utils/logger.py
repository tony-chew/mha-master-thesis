import torch
import utils
import random
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

MODELS_EXT = "pth"


class Logger:
    """
    Logging of various performance metrics
    """
    def __init__(self,
                 log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_log(self,
                   tag,
                   value,
                   epoch):
        self.writer.add_scalar(tag, value, epoch)

    def image_log(self,
                  tag,
                  img):
        self.writer.add_image(tag, img)

    def figure_log(self,
                   tag,
                   fig):
        self.writer.add_figure(tag, fig)

    def close(self):
        self.writer.close()


class AverageTracker:
    """
    Compute and store the average and current value of the loss
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self,
               value,
               n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Keep track of the behaviour of the validation loss for early stopping
    """
    def __init__(self,
                 patience,
                 mode=None,
                 num_bad_epochs=0,
                 min_delta=0,
                 best_val=100,
                 best_miou=-100,
                 last_valid_epoch=None,
                 last_valid_epoch_results=None,
                 last_valid_hyp_results=None,
                 last_valid_model=None):
        """
        :param num_bad_epochs: number of epochs in a row which did not meet validation criteria
        :param min_delta: the allowable difference for the validation loss between epochs to continue training on
        :param best_val: the best validation loss to compare the current epoch results to
        :param best_miou: the best mIoU to compare the current epoch results to
        :param last_valid_epoch: if checkpointing, the last valid epoch is loaded here
        :param last_valid_epoch_results: if checkpointing, the results of the last valid epoch is loaded here
        :param last_valid_hyp_results: if checkpointing, the hyp results of the last valid epoch is loaded here
        :param last_valid_model: if checkpointing, the model of the last valid epoch is loaded here
        :param patience: the number of allowable epochs with increasing validation loss (in a row) before stopping
                            training
        """
        logging.basicConfig(level=logging.INFO)
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.num_bad_epochs = num_bad_epochs
        self.best_val = best_val
        self.best_miou = best_miou
        self.last_valid_epoch = last_valid_epoch
        self.last_valid_epoch_results = last_valid_epoch_results
        self.last_valid_hyp_results = last_valid_hyp_results
        self.last_valid_model = last_valid_model

    def step(self, val, mean_iou=None):
        """
        Determine the conditions for stopping training. If the given val loss/mean iou is higher/lower than the best
        val loss, then add one to the patience counter. Reset if val loss/mean iou is lower/higher than the best
        val loss/mean iou. Stop training when the patience counter reaches desired epochs (thus meaning val loss/mean
        iou has not improved for x amount of epochs. Abandon this training run
        """
        if self.criteria(val, mean_iou):
            self.num_bad_epochs = 0
            self.best_val = val

            if self.mode == 'semseg' or self.mode == 'mha':
                self.best_miou = mean_iou

        else:
            self.num_bad_epochs += 1
            logging.info(f"{self.num_bad_epochs} epochs in a row where val loss did not improve from previous epoch")

        return self.num_bad_epochs, self.best_val, self.best_miou

    def track_last_epoch(self,
                         model_path,
                         model,
                         epoch_count,
                         results,
                         params,
                         param_path,
                         rank,
                         hyp_results=None,
                         truth_tensor=None):
        """
        Keep track of the last valid epoch and its corresponding results (when num_bad_epochs goes back to 0). Save the
        results of the last valid epoch and its model when the patience limit is reached (when num_bad_epochs is larger
        than or equal to the patience.)

        :param model_path: the savepath for the model
        :param model: the model to be saved
        :param epoch_count: the current epoch
        :param results: the results of the current epoch
        :param hyp_results: the hyp run results
        :param params: the parameters of that hyperparameter run
        :param param_path: the savepath for the parameters
        :param rank: the current rank at which the process is running on
        :param truth_tensor: the truth tensor that will eventually broadcast to the other ranks
        """
        if self.num_bad_epochs == 0:
            self.last_valid_epoch = epoch_count
            self.last_valid_epoch_results = results
            self.last_valid_hyp_results = hyp_results
            self.last_valid_model = model

        if self.num_bad_epochs >= self.patience:
            save_checkpoint(target_path=model_path,
                            epoch_count=self.last_valid_epoch,
                            model=self.last_valid_model,
                            results=self.last_valid_epoch_results,
                            hyp_results=self.last_valid_hyp_results,
                            early_stopping=True)

            utils.save_contents(contents=params,
                                target_path=param_path,
                                epoch_results=results,
                                mode=self.mode)

            truth_tensor = torch.tensor([True]).to(rank)

        return truth_tensor, self.last_valid_epoch, self.last_valid_epoch_results, \
               self.last_valid_hyp_results, self.last_valid_model

    def criteria(self, val, mean_iou):
        if self.mode == 'autoencoder' or self.mode == 'mha':
            return val < self.best_val - self.min_delta
        elif self.mode == 'semseg':
            return mean_iou > self.best_miou + self.min_delta


def save_checkpoint(target_path,
                    epoch_count,
                    model,
                    results,
                    hyp_results=None,
                    hyp_run=None,
                    params=None,
                    model_params=None,
                    optimiser=None,
                    scheduler=None,
                    num_bad_epochs=None,
                    best_val_epochs=None,
                    last_valid_epoch=None,
                    last_valid_epoch_results=None,
                    last_valid_hyp_results=None,
                    last_valid_model=None,
                    best_val=None,
                    early_stopping=False,
                    best_mean_iou=None,
                    best_class_iou=None):
    """
    Save the checkpoint including the model, parameters and current training information

    :param target_path: filepath of the model to be saved to
    :param epoch_count: current epoch to be saved
    :param model: current model to be saved
    :param optimiser: current optimiser to be saved
    :param scheduler: the scheduler to be saved
    :param results: current results of the training to be saved
    :param hyp_run: if hyperparameter search is set, the current run to be saved
    :param params: current parameters of the training run to be saved
    :param model_params: current model parameters to be saved
    :param hyp_results: dictionary of the results of each hyperparameter run
    :param num_bad_epochs: tally of number of epochs in a row where validation loss was not higher than best val
    :param best_val_epochs: the tracker for the best validation loss of that run
    :param last_valid_epoch: the last epoch number in which the results are valid
    :param last_valid_epoch_results: the last epoch results in which the results are valid
    :param last_valid_hyp_results: the last hyp run in which the results are valid
    :param last_valid_model: the last model in which the results are valid
    :param best_val: tracker of the best validation loss of the hyperparameter runs
    :param early_stopping: tag to save checkpoint under early stopping conditions (only the model and results are saved)
    :param best_mean_iou: the best mean iou to save
    :param best_class_iou: the best class iou to save
    """
    state = {
        "epoch": epoch_count,
        "model_state": utils.save_parallelised_model(model),
        "optimiser_state": optimiser.state_dict() if optimiser is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "results_state": results,
        "parameters": params,
        "hyperparameter_run": hyp_run,
        "model_parameters": model_params,
        "hyp_results": hyp_results,
        "num_bad_epochs": num_bad_epochs,
        "best_val_epochs": best_val_epochs,
        "last_valid_epoch": last_valid_epoch,
        "last_valid_epoch_results": last_valid_epoch_results,
        "last_valid_hyp_results": last_valid_hyp_results,
        "last_valid_model_state": utils.save_parallelised_model(last_valid_model) if last_valid_model is not None else None,
        "best_val_hyp": best_val,
        "best_mean_iou": best_mean_iou,
        "best_class_iou": best_class_iou,
        "rand_state": random.getstate(),
        "np_rand_state": np.random.get_state(),
        "torch_rand_state": torch.get_rng_state(),
        "torch.cuda_rand_state": torch.cuda.get_rng_state()
    }
    model_path = target_path / f"early_stopping_checkpoint.{MODELS_EXT}" if early_stopping \
        else target_path / f"checkpoint.{MODELS_EXT}"
    torch.save(state, model_path)


def reset_counters():
    """
    Bring the counters back to its initial state
    """
    epoch_count = 0
    results = []

    return epoch_count, results





