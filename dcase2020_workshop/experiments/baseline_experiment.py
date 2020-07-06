from dcase2020_workshop.experiments import BaseExperiment
from dcase2020_workshop.utils.logger import Logger
from datetime import datetime
import os
import pytorch_lightning as pl
from sacred import Experiment

from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'
from dcase2020_workshop.data_sets import AudioSet, ComplementMCMDataSet, DATA_PATH


class BaselineExperiment(BaseExperiment, pl.LightningModule):

    '''
    DCASE Baseline with AE, MADMOG & MAF per machine ID.
    '''

    def __init__(self, configuration_dict, _run):
        super().__init__(configuration_dict)

        self.network = self.objects['model']
        self.reconstruction = self.objects['reconstruction']
        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)

        # will be set before each epoch
        self.normal_data_set = self.objects['normal_data_set']

        # experiment state variables
        self.epoch = -1
        self.step = 0
        self.result = None

    def forward(self, batch):
        batch['epoch'] = self.epoch
        batch = self.network(self.normalize_batch(batch))
        return batch

    def training_step(self, batch_normal, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            batch_normal = self(batch_normal)

            if batch_normal.get('prior_loss'):
                batch_normal['loss'] = batch_normal['reconstruction_loss'] + batch_normal['prior_loss']
            else:
                batch_normal['loss'] = batch_normal['reconstruction_loss']

            self.logger_.log_training_step(batch_normal, self.step)
            self.step += 1
        else:
            raise AttributeError

        return {
            'loss': batch_normal['loss'],
            'tqdm': {'loss': batch_normal['loss']},
        }

    def validation_step(self, batch, batch_num):
        self(batch)
        if batch_num == 0:
            self.logger_.log_image_reconstruction(batch, self.epoch)
        return {
            'targets': batch['targets'],
            'scores': batch['scores'],
            'machine_types': batch['machine_types'],
            'machine_ids': batch['machine_ids'],
            'file_ids': batch['file_ids']
        }

    def validation_epoch_end(self, outputs):
        self.logger_.log_validation(outputs, self.step, self.epoch)
        return {}

    def test_step(self, batch, batch_num):
        return self.validation_step(batch, batch_num)

    def test_epoch_end(self, outputs):
        self.result = self.logger_.log_test(outputs)
        self.logger_.close()
        return self.result


def configuration():

    #####################
    # configuration, uses default parameters of more detailed configuration
    #####################

    id = None
    seed = 1220
    deterministic = False
    log_path = os.path.join('experiment_logs', id)
    data_path = DATA_PATH

    machine_type = 0
    machine_id = 0

    normal_data_set_class = 'dcase2020_workshop.data_sets.MCMDataSet'
    prior_class = 'dcase2020_workshop.priors.NoPrior'
    latent_size = 8

    feature_settings = {
        'data_path': data_path,
        'num_mel': 128,
        'n_fft': 1024,
        'hop_size': 512,
        'power': 2.0,
        'fmin': 0,
        'fmax': None,
        'normalize_raw': 'unit_variance',
        'normalize_spec': None,
        'window_length': 5,
        'window_hop_size': 1
    }

    training_settings = {
        'epochs': 100,
        'batch_size': 512,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'learning_rate_decay': 1.0
    }

    debug = False
    if debug:
        num_workers = 0
    else:
        num_workers = 4

    ########################
    # automatic object creation
    ########################

    normal_data_set = {
        'class': normal_data_set_class,
        'kwargs': {
            'machine_type': machine_type,
            'machine_id': machine_id,
            **feature_settings
        }
    }

    prior = {
        'class': prior_class,
        'kwargs': {
            'latent_size': latent_size,
            'weight': 1
        }
    }

    reconstruction = {
        'class': 'dcase2020_workshop.losses.MSEReconstruction',
        'kwargs': {
            'weight': 1,
            'input_shape': '@normal_data_set.observation_shape'
        }
    }

    model = {
        'class': 'dcase2020_workshop.models.AE',
        'args': [
            '@normal_data_set.observation_shape',
            '@reconstruction',
            '@prior'
        ]
    }

    lr_scheduler = {
        'class': 'torch.optim.lr_scheduler.ExponentialLR',
        'args': [
            '@optimizer',
        ],
        'kwargs': {
            'gamma': training_settings['learning_rate_decay']
        }
    }

    optimizer = {
        'class': 'torch.optim.Adam',
        'args': [
            '@model.parameters()'
        ],
        'kwargs': {
            'lr': training_settings['learning_rate'],
            'betas': (0.9, 0.999),
            'amsgrad': False,
            'weight_decay': training_settings['weight_decay'],
        }
    }

    trainer = {
        'class': 'dcase2020_workshop.trainers.PTLTrainer',
        'kwargs': {
            'max_epochs': training_settings['epochs'],
            'checkpoint_callback': False,
            'logger': False,
            'early_stop_callback': False,
            'gpus': [0],
            'show_progress_bar': True,
            'progress_bar_refresh_rate': 1000
        }
    }


ex = Experiment('dcase2020_workshop_BaselineExperiment')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = BaselineExperiment(_config, _run)
    return experiment.run()
