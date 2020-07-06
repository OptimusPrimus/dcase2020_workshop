from dcase2020_workshop.losses import BaseReconstruction, BaseLoss
import torch
import torch.nn.functional as F


class AUC(BaseLoss):

    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, batch_normal):

        assert batch_normal.get('scores') is not None, "cannot compute loss without scores"

        normal_scores = batch_normal['scores'][batch_normal['outlier'] == 0]
        outlier_scores = batch_normal['scores'][batch_normal['outlier'] == 1]

        tprs = torch.sigmoid(outlier_scores[:, None] - normal_scores[None, :]).mean(dim=0)
        batch_normal['tpr'] = tprs.mean()
        batch_normal['fpr'] = 0.5

        batch_normal['loss_raw'] = - batch_normal['tpr']
        batch_normal['loss'] = self.weight * batch_normal['loss_raw']

        # log some stuff...
        batch_normal['normal_scores_mean'] = normal_scores.mean()
        batch_normal['normal_scores_std'] = normal_scores.std()
        batch_normal['outlier_scores_mean'] = outlier_scores.mean()
        batch_normal['outlier_scores_std'] = outlier_scores.std()

        return batch_normal
