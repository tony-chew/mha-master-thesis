import torch
import collections


MultiheadWeights = collections.namedtuple(
    "MultiheadWeights",
    "ae_weight semseg_weight")


class MultiHeadLoss(torch.nn.Module):
    """
    Establish the total loss for the multi-head training setup
    """
    def __init__(self,
                 losses,
                 loss_weights):
        """
        :param losses: list of losses corresponding to the autoencoder and segmentation loss individually
        :param loss_weights: the weights for each invdividual head to apply the loss to
        """
        super(MultiHeadLoss, self).__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.loss_weights = MultiheadWeights(loss_weights[0], loss_weights[1])

    def forward(self,
                head_preds,
                head_targets,
                train_mode="endtoend",
                iteration_count=None):
        """
        Conduct the forward function to get the multi-head losses

        :param head_preds: the prediction result of each head
        :param head_targets: the target result of each head
        :param train_mode: mode of training -> in "stepbystep" or "endtoend" mode. Sets the according loss to zero
        :param iteration_count: the iteration count if in train mode "stepbystep"
        :return total_loss: the total loss of both heads
        :return head_losses: the loss of each individual head
        """
        total_loss, head_losses = self._forward_imp(head_preds,
                                                    head_targets,
                                                    train_mode,
                                                    iteration_count)

        return total_loss, head_losses

    def _forward_imp(self,
                     head_preds,
                     head_targets,
                     train_mode,
                     iteration_count):
        """
        Calculate the loss of each head, then apply their corresponding weight, then add together to get the final total
        loss

        :param head_preds: a list of the predictions (autoencoder_pred, segmentation_pred)
        :param head_targets: a list of the targets (autoencoder_target, segmentation_target)
        :param train_mode: mode of training -> in "stepbystep" or "endtoend" mode. Sets the according loss to zero
        :param iteration_count: the iteration count if in train mode "stepbystep"
        """
        ae_loss, semseg_loss = self.losses

        # Get autoencoder loss
        ae_pred = head_preds[0]
        ae_gt = head_targets[0]
        ae_loss_metric = ae_loss(ae_pred, ae_gt)

        # Get segmentation loss
        semseg_pred = head_preds[1]
        semseg_gt = head_targets[1]
        semseg_loss_metric = semseg_loss(semseg_pred, semseg_gt)

        # Apply weights
        ae_loss_metric *= self.loss_weights.ae_weight
        semseg_loss_metric *= self.loss_weights.semseg_weight

        if train_mode == 'stepbystep' and iteration_count % 2 == 0:
            # Train autoencoder only
            semseg_loss_metric = 0 * semseg_loss_metric
        elif train_mode == 'stepbystep' and iteration_count % 2 != 0:
            # Train segmentation only
            ae_loss_metric = 0 * ae_loss_metric

        loss = ae_loss_metric + semseg_loss_metric

        return loss, (ae_loss_metric.item(), semseg_loss_metric.item())

