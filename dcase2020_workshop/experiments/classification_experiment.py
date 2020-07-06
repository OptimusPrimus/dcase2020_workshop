from dcase2020_workshop.experiments import BaseExperiment
import pytorch_lightning as pl
import torch
from sacred import Experiment
from dcase2020_workshop.utils.logger import Logger
import os
import torch.utils.data
# workaround...
from sacred import SETTINGS

SETTINGS['CAPTURE_MODE'] = 'sys'
from dcase2020_workshop.data_sets import DATA_PATH


class ClassificationExperiment(BaseExperiment, pl.LightningModule):

    def __init__(self, configuration_dict, _run):
        super().__init__(configuration_dict)

        # default stuff
        self.network = self.objects['model']
        self.loss = self.objects['loss']
        self.logger_ = Logger(_run, self, self.configuration_dict, self.objects)

        # will be set before each epoch
        self.normal_data_set = self.objects['normal_data_set']
        self.outlier_data_set = self.objects['outlier_data_set']

        if self.outlier_data_set is not None:
            self.inf_data_loader = self.get_inf_data_loader(
                torch.utils.data.DataLoader(
                    self.outlier_data_set.training_data_set(),
                    batch_size=self.objects['training_settings']['batch_size'],
                    shuffle=True,
                    num_workers=self.objects['num_workers'],
                    drop_last=False
                )
            )

        # experiment state variables
        self.epoch = -1
        self.step = 0
        self.result = None

    def get_inf_data_loader(self, dl):
        device = next(iter(self.network.parameters())).device
        while True:
            for batch in iter(dl):
                for key in batch:
                    if type(batch[key]) is torch.Tensor:
                        batch[key] = batch[key].to(device)
                yield batch

    def forward(self, batch):
        batch['epoch'] = self.epoch
        batch = self.network(self.normalize_batch(batch))
        return batch

    def training_step(self, normal_batch, batch_num, optimizer_idx=0):

        if batch_num == 0 and optimizer_idx == 0:
            self.epoch += 1

        if optimizer_idx == 0:
            outlier_batch = next(self.inf_data_loader)

            normal_batch_size = len(normal_batch['observations'])
            outlier_batch_size = len(outlier_batch['observations'])

            device = normal_batch['observations'].device

            normal_batch['outlier'] = torch.cat([
                torch.zeros(normal_batch_size, 1).to(device),
                torch.ones(outlier_batch_size, 1).to(device)
            ])

            normal_batch['observations'] = torch.cat([
                normal_batch['observations'],
                outlier_batch['observations']
            ])

            normal_batch = self(normal_batch)

            normal_batch = self.loss(normal_batch)

            self.logger_.log_training_step(normal_batch, self.step)
            self.step += 1
        else:
            raise AttributeError

        return {
            'loss': normal_batch['loss'],
            'tqdm': {'loss': normal_batch['loss']},
        }

    def validation_step(self, batch, batch_num):
        self(batch)
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
        self.logger_.__log_model__()
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
    outlier_data_set_class = 'dcase2020_workshop.data_sets.ComplementMCMDataSet'

    blur=None

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
        'window_length': 256,
        'window_hop_size': None
    }

    outlier_settings = {
        'valid_types': 'same_mic_all_types',
        'machine_type': machine_type,
        'machine_id': machine_id,
        'num_samples': None,
        'num_classes': None,
        'blur': blur
    }

    model_settings = {
        'base_channels': 64,
        'num_outputs': 1,
        'rf': 'normal',
        'dropout_probability': 0.0
    }

    training_settings = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'learning_rate_decay': 0.98
    }

    debug = False
    if debug:
        num_workers = 0
        training_settings['epochs'] = 1
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

    outlier_data_set = {
        'class': outlier_data_set_class,
        'kwargs': {
            **feature_settings,
            **outlier_settings
        }
    }

    model = {
        'class': 'dcase2020_workshop.models.ResNet',
        'args': [
            '@normal_data_set.observation_shape'
        ],
        'kwargs': model_settings
    }

    loss = {
        'class': 'dcase2020_workshop.losses.BCE',
        'kwargs': {
            'weight': 1.0,
            'input_shape': '@normal_data_set.observation_shape'
        }
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


ex = Experiment('dcase2020_workshop_ClassificationExperiment')
cfg = ex.config(configuration)


@ex.automain
def run(_config, _run):
    experiment = ClassificationExperiment(_config, _run)
    return experiment.run()
