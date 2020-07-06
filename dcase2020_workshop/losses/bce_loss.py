from dcase2020_workshop.losses import BaseReconstruction, BaseLoss
import torch
import torch.nn.functional as F


class BCE(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):

        super().__init__()
        self.weight = weight

    def forward(self, batch_normal):

        assert batch_normal.get('scores') is not None, "cannot compute loss without scores"

        normal_scores = batch_normal['scores'][batch_normal['outlier'] == 0]
        outlier_scores = batch_normal['scores'][batch_normal['outlier'] == 1]

        loss = F.binary_cross_entropy_with_logits(
            batch_normal['scores'],
            batch_normal['outlier']
        )

        batch_normal['loss_raw'] = loss
        batch_normal['loss'] = self.weight * batch_normal['loss_raw']

        # log some stuff...
        batch_normal['normal_scores_mean'] = normal_scores.mean()
        batch_normal['normal_scores_std'] = normal_scores.std()
        batch_normal['outlier_scores_mean'] = outlier_scores.mean()
        batch_normal['outlier_scores_std'] = outlier_scores.std()

        return batch_normal
