#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
import utils
import transforms
import numpy as np
from PIL import Image
import torchvision.transforms as tf
from sklearn.manifold import TSNE


class Visualise:
    """
    Prepare the visualisation of the results
    """
    def __init__(self,
                 dataloader,
                 num_images,
                 device):
        """
        :param dataloader: the dataset
        :param num_images: number of images to display
        :param device: device to compute on
        """
        self.num_images = num_images
        self.xs_disp = torch.tensor(()).to(device)
        self.ps_disp = torch.tensor(()).to(device)
        # self.heat_map = torch.zeros(utils.image_shape(dataloader)).to(device)

    def update(self,
               xs,
               ps):
        """
        Update the visual tool with the next image or reconstruction
        """
        if len(self.xs_disp) != self.num_images:
            self.xs_disp = torch.cat((xs, self.xs_disp), 0)
            self.ps_disp = torch.cat((ps, self.ps_disp), 0)

        # self.heat_map = torch.add(self.heat_map, utils.recons_error_map(xs, ps))

    def create_visuals(self,
                       writer,
                       type,
                       hyp_tag=False,
                       hyp_run=None):
        """
        Create the visual display and output to Tensorboard
        """
        self.xs_disp, self.ps_disp = self.xs_disp.to("cpu"), self.ps_disp.to("cpu")
        # err = utils.recons_error_map(self.xs_disp, self.ps_disp)
        # visualise_test_results = utils.stack_images((self.xs_disp, self.ps_disp, err))
        visualise_test_results = utils.stack_images((self.xs_disp, self.ps_disp))
        mse = utils.display_images(visualise_test_results, self.num_images)
        # heat_map_norm = utils.display_images((utils.norm_zero_one(self.heat_map)))

        if type == "test":
            writer.image_log(f"Cityscapes {type} images: Reconstruction and Reconstruction Error (Tensorboard)",
                             mse)
            # writer.image_log(f"Heat map of Cityscapes {type} set",
            #                  heat_map_norm)
        elif type == "val":
            if hyp_tag:
                writer.image_log(f"Cityscapes {type} images: Reconstruction and Reconstruction Error (Tensorboard) "
                                 f"for hyperparameter run {hyp_run}",
                                 mse)
                # writer.image_log(f"Heat map of Cityscapes {type} set for hyperparameter run {hyp_run}",
                #                  heat_map_norm)
            else:
                writer.image_log(f"Cityscapes {type} images: Reconstruction and Reconstruction Error (Tensorboard)",
                                 mse)
                # writer.image_log(f"Heat map of Cityscapes {type} set",
                #                  heat_map_norm)


class VisualiseSemSeg:
    """
    Prepare the visualisation of the Semantic Segmentation results
    """
    def __init__(self,
                 num_images):
        """
        :param num_images: number of images to display
        """
        self.num_images = num_images
        self.xs = []
        self.mask = []
        self.gt = []

    def update(self,
               xs,
               mask,
               gt=None):
        """
        Update the visual tool with the next image or prediction
        """
        if len(self.xs) != self.num_images:
            xs = torch.squeeze(xs, dim=0)
            xs = tf.transforms.ToPILImage()(xs)
            if gt is not None:
                gt = torch.squeeze(gt, dim=0)
                gt = tf.transforms.ToPILImage()(gt)
                self.gt.append(gt)

            self.xs.append(xs)
            self.mask.append(mask)

    def create_visuals(self,
                       writer,
                       save_path):
        """
        Create the visual display and save it to the stated save path
        """
        semseg_images = utils.semseg_hstack(self.xs)
        semseg_labels = utils.semseg_hstack(self.mask)
        if len(self.gt) == self.num_images:
            semseg_gt = utils.semseg_hstack(self.gt)
            semseg_out = [semseg_images, semseg_gt, semseg_labels]
        else:
            semseg_out = [semseg_images, semseg_labels]

        semseg_out = utils.semseg_vstack(semseg_out)
        semseg_out.save(save_path / "semseg_val.png")


class VisualiseLatentSpace():
    """
    Visualise the latent space of the autoencoder
    """
    def __init__(self,
                 model,
                 mean,
                 std,
                 num_samples):
        """
        :param model: the autoencoder model
        :param mean: mean of the dataset
        :param std: std of the dataset
        :num_samples: number of samples to show in the latent space visualisation
        """
        self.model = model.eval()
        self.mean = mean
        self.std = std

        self.data = []
        self.targets = []
        self.counter = 0
        self.num_samples = num_samples
        self.to_image = tf.ToPILImage()

    def update(self,
               xs):
        """
        Update the list with the image and its target. Check if number of samples is surpassed first

        :param xs: the target
        """
        if self.counter_check():
            output = self.model.encoder_bottleneck(xs)

            xs = transforms.inverse_normalise(tensor=xs,
                                              mean=torch.tensor(tuple(self.mean)),
                                              std=torch.tensor(tuple(self.std)))
            xs = xs.detach().cpu()
            output = output.detach().cpu().numpy()

            for output, xs in zip(output, xs):
                self.data.append(output.reshape(512))
                self.targets.append(xs)
                self.counter += 1

    def counter_check(self):
        """
        A basic check for the number of samples already listed
        """
        return self.counter <= self.num_samples

    def visualise(self,
                  save_path):
        """
        Construct the visualisation and save to save_path
        """
        data = np.array(self.data)
        targets = np.array(self.targets, dtype=object)

        data = data[:int(len(data) * 0.5)]
        targets = targets[:int(len(targets) * 0.5)]

        tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)
        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        width = 8000
        height = 7000

        full_image = Image.new('RGB', (width, height), (255, 255, 255))
        for img, x, y in zip(targets, tx, ty):
            tile = self.to_image(img)
            full_image.paste(tile, (int((width - 1200) * (x)), int((height - 500) * (y))), mask=tile.convert('RGBA'))

        plt.figure(figsize=(240, 240))
        plt.imshow(full_image)
        plt.axis("off")
        # plt.show()
        full_image.save(save_path / "autoencoder_latent_vis.png")


def visualise_segmentation_features(model,
                                    xs,
                                    save_path):
    """
    Visualise the features of the weights from three specific points of the semantic segmentation model
    """
    model.eval()

    # Acquire the hooks
    activation = {}
    model.encoder.encoder[0][0].conv.conv.register_forward_hook(utils.get_activation('conv1',
                                                                                     activation))
    model.encoder.encoder[1][5].conv3.rw_conv.conv.register_forward_hook(utils.get_activation('conv2',
                                                                                              activation))
    model.encoder.encoder[2][4].conv3.rw_conv.conv.register_forward_hook(utils.get_activation('conv3',
                                                                                              activation))
    output = model(xs)
    output_1 = activation['conv1'].squeeze().cpu()
    output_2 = activation['conv2'].squeeze().cpu()
    output_3 = activation['conv3'].squeeze().cpu()

    # Plot the three hooks
    fig1 = plt.figure(figsize=(30, 20.1))
    ax1 = [fig1.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
    for i, a in enumerate(ax1):
        a.imshow(output_1[i], 'gray')
        a.axis("off")
    fig1.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path / "output_one", bbox_inches='tight', pad_inches=0, transparent=True)

    fig2 = plt.figure(figsize=(15, 10.05))
    ax2 = [fig2.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
    for i, a in enumerate(ax2):
        a.imshow(output_2[i], 'gray')
        a.axis("off")
    fig2.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path / "output_two", bbox_inches='tight', pad_inches=0, transparent=True)

    fig3 = plt.figure(figsize=(15, 10.05))
    ax3 = [fig3.add_subplot(4, 3, i + 1, adjustable='box', aspect=0.5) for i in range(12)]
    for i, a in enumerate(ax3):
        a.imshow(output_3[i], 'gray')
        a.axis("off")
    fig3.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path / "output_three", bbox_inches='tight', pad_inches=0, transparent=True)


def plot_training_vs_validation_vs_accuracy(results,
                                            mode,
                                            figsize=None):
    """
    Plot a graph with epochs vs training loss and validation loss.

    :param results: results of the training run depending on the mode
    :param mode: autoencoder, semseg or mha mode
    :param figsize: figure size of the plot
    """
    if mode == 'autoencoder':
        epochs, _, trainings, validations = zip(*results)
    elif mode == 'semseg':
        epochs, _, trainings, validations, _ = zip(*results)
    elif mode == 'mha':
        epochs, _, trainings, validations, ae_val_loss, ss_val_loss, ae_train_loss, ss_train_loss, _ = zip(*results)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # training loss vs. validation loss
    tax = ax
    tax.set_title('losses vs epoch')
    tax.set_xlabel("epoch")
    tax.set_ylabel("training loss")
    tax.plot(epochs, trainings, color="red")
    tax.tick_params(axis="y", labelcolor="red")

    vax = tax.twinx()
    vax.set_ylabel("validation loss")
    vax.plot(epochs, validations, color="blue")
    vax.tick_params(axis="y", labelcolor="blue")
    fig.tight_layout()

    if mode == 'mha':
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ae = ax2
        ae.set_title('autoencoder losses vs epoch')
        ae.set_xlabel("epoch")
        ae.set_ylabel("autoencoder validation loss")
        ae.plot(epochs, ae_val_loss, color="red")
        ae.tick_params(axis="y", labelcolor="red")

        ae2 = ae.twinx()
        ae2.set_ylabel("autoencoder training loss")
        ae2.plot(epochs, ae_train_loss, color="blue")
        ae2.tick_params(axis="y", labelcolor="blue")
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ss_val = ax3
        ss_val.set_title('segmentation losses vs epoch')
        ss_val.set_xlabel("epoch")
        ss_val.set_ylabel("segmentation validation loss")
        ss_val.plot(epochs, ss_val_loss, color="red")
        ss_val.tick_params(axis="y", labelcolor="red")

        ss_train = ss_val.twinx()
        ss_train.set_ylabel("segmentation training loss")
        ss_train.plot(epochs, ss_train_loss, color="blue")
        ss_train.tick_params(axis="y", labelcolor="blue")
        fig3.tight_layout()

        return fig, fig2, fig3

    return fig


def plot_iou_vs_epoch(results,
                      mode,
                      figsize=None):
    """
    Plot the miou against epoch
    """
    if mode == 'mha':
        epochs, _, _, _, _, _, _, _, mean_iou = zip(*results)
    else:
        epochs, _, _, _, mean_iou = zip(*results)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    iou = ax
    iou.set_title('mIoU vs epoch')
    iou.set_xlabel('epoch')
    iou.set_ylabel('mIoU')
    iou.plot(epochs, mean_iou)

    return fig


def plot_hyp_runs(hyp_run_results,
                  figsize=None):
    """
    Plot a graph of training and validation loss for each hyperparameter run

    :param hyp_run_results: hyp_run: epoch, training loss, validation loss
    :param figsize: the size of the figures
    """
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    tax = ax1
    tax.set_title("Training loss vs epochs")
    tax.set_xlabel("epoch")
    tax.set_ylabel("training loss")

    vax = ax2
    vax.set_title("Validation loss vs epochs")
    vax.set_xlabel("epoch")
    vax.set_ylabel("validation loss")

    for hyp_run, results in hyp_run_results.items():
        epochs, best_fps, trainings, validations = zip(*results)

        tax.plot(epochs, trainings, label=f"hyperparam: {hyp_run}")
        vax.plot(epochs, validations, label=f"hyperparam: {hyp_run}")

    tax.legend(loc="upper right")
    vax.legend(loc="upper right")

    return fig1, fig2
