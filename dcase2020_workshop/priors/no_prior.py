from dcase2020_workshop.priors import PriorBase
import torch.nn


class NoPrior(PriorBase):

    def __init__(self, latent_size=256, **kwargs):
        super().__init__()
        self.latent_size_ = latent_size

    def forward(self, batch):

        """ No modification """
        batch['codes'] = batch['pre_codes']
        batch['prior_loss'] = torch.tensor(0.0)

        # log some stuff...
        batch['mean_pre_codes'] = batch['prior_loss'].mean()
        batch['std_pre_codes'] = batch['prior_loss'].std()

        return batch

    @property
    def latent_size(self):
        return self.latent_size_

    @property
    def input_size(self):
        return self.latent_size_
