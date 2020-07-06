import os
import torch.utils.data
import glob
from dcase2020_workshop.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, EVALUATION_ID_MAP, \
    ALL_ID_MAP, \
    enumerate_development_datasets, enumerate_evaluation_datasets

import numpy as np
from dcase2020_workshop.data_sets.audio_processor import __load_preprocess_file__
from concurrent.futures import ThreadPoolExecutor
import pickle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


class MCMDataSet(BaseDataSet):

    def __init__(
            self,
            **kwargs
    ):
        assert type(kwargs['machine_type']) == int and type(kwargs['machine_id']) == int
        assert kwargs['machine_id'] >= 0
        assert kwargs['machine_type'] >= 0

        kwargs['data_path'] = os.path.join(kwargs['data_path'], 'dcase2020_task2')
        self.kwargs = kwargs

        self.training_set = MachineDataSet(mode='training', **kwargs)
        self.validation_set = MachineDataSet(mode='validation', **kwargs)

    @property
    def observation_shape(self) -> tuple:
        return 1, self.kwargs['num_mel'], self.kwargs['window_length']

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        return self.validation_set


class MachineDataSet(torch.utils.data.Dataset):

    def __init__(
            self,
            machine_type=-1,
            machine_id=-1,
            mode='training',
            data_path=None,

            sr=16000,
            num_mel=128,
            n_fft=1024,
            hop_size=512,
            power=2.0,
            fmin=0,
            fmax=None,
            normalize_raw='unit_variance',
            normalize_spec=None,

            blur=None,

            window_length=256,
            window_hop_size=None


    ):

        assert mode in ['training', 'validation']
        if mode == 'validation':
            window_hop_size = 1

        self.data_path = data_path
        self.machine_type = INVERSE_CLASS_MAP[machine_type]
        self.machine_id = machine_id
        self.mode = mode

        self.audio_features = {
            'sr': sr,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'power': power,
            'fmin': fmin,
            'fmax': fmax,
            'normalize_raw': normalize_raw,
            'normalize_spec': normalize_spec,
            'blur': blur
        }

        self.window_length = window_length
        self.window_hop_size = window_hop_size

        self.data, self.meta_data = self.__load_data__()

        self.index_map = {}
        ctr = 0
        for i, file in enumerate(self.data):
            if window_hop_size is None:
                self.index_map[ctr] = (i, file.shape[-1] - window_length)
                ctr += 1
            else:
                for j in range(file.shape[-1] + 1 - window_length):
                    if j % window_hop_size == 0:
                        self.index_map[ctr] = (i, j)
                        ctr += 1
        self.length = ctr

    def __getitem__(self, item):
        file_idx, offset = self.index_map[item]
        if self.window_hop_size is None:
            offset = np.random.randint(0, offset)
        observation = self.data[file_idx][:, offset:offset + self.window_length]
        meta_data = self.meta_data[file_idx].copy()

        if self.audio_features.get('blur'):
            sigma = self.audio_features.get('blur')
            assert sigma > 0
            meta_data['observations'] = gaussian_filter(observation, sigma=sigma)[None]
        else:
            meta_data['observations'] = observation[None]

        return meta_data

    def __len__(self):
        return self.length

    def __load_data__(self):

        file_id = '{sr}_{num_mel}_{n_fft}_{hop_size}_{power}_{fmin}_{fmax}_{normalize_raw}_{normalize_spec}_{machine_type}_{machine_id}_{mode}'.format(
            machine_type=self.machine_type,
            machine_id=self.machine_id,
            mode=self.mode,
            **self.audio_features
        )
        data_path = os.path.join(self.data_path, file_id + ".npz")
        meat_data_path = os.path.join(self.data_path, file_id + ".pkl")

        data = []
        meta_data = []
        if os.path.exists(data_path):
            print('Loading {} data set for machine type {} id {}...'.format(
                self.mode, self.machine_type, self.machine_id
            ))
            container = np.load(data_path)
            data = [container[key] for key in container]
            with open(meat_data_path, "rb") as f:
                meta_data = pickle.load(f)
        else:
            print('Loading & Saving {} data set for machine type {} id {}...'.format(
                self.mode, self.machine_type, self.machine_id
            ))

            files = self.__get_file_list__()

            def closure(f):
                return __load_preprocess_file__(
                    f,
                    config=self.audio_features,
                    trim_zeros=False,
                    min_size=None,
                    max_duration=None
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                for f, r in tqdm(zip(files, executor.map(closure, files)), total=len(files)):
                    if r is not None:
                        data.append(r)
                        meta_data.append(self.__get_meta_data__(f))

            np.savez(data_path, *data)
            with open(meat_data_path, "wb") as f:
                pickle.dump(meta_data, f)

        return data, meta_data

    def __get_meta_data__(self, file_path):
        meta_data = os.path.split(file_path)[-1].split('_')
        machine_type = os.path.split(os.path.split(os.path.split(file_path)[0])[0])[1]
        assert self.machine_type == machine_type
        machine_type = CLASS_MAP[machine_type]

        if len(meta_data) == 4:
            if meta_data[0] == 'normal':
                y = 0
            elif meta_data[0] == 'anomaly':
                y = 1
            else:
                raise AttributeError
            assert self.machine_id == int(meta_data[2])
        elif len(meta_data) == 3:
            y = -1
            assert self.machine_id == int(meta_data[1])
        else:
            raise AttributeError

        return {
            'targets': y,
            'machine_types': machine_type,
            'machine_ids': self.machine_id,
            'file_ids': os.sep.join(os.path.normpath(file_path).split(os.sep)[-4:])
        }

    def __get_file_list__(self):

        if self.machine_id in TRAINING_ID_MAP[CLASS_MAP[self.machine_type]]:
            root_folder = 'dev_data'
        elif self.machine_id in EVALUATION_ID_MAP[CLASS_MAP[self.machine_type]]:
            root_folder = 'eval_data'
        else:
            raise AttributeError

        if self.mode == 'training':
            files = glob.glob(
                os.path.join(
                    self.data_path, root_folder, self.machine_type, 'train',
                    '*id_{:02d}_*.wav'.format(self.machine_id)
                )
            )
        elif self.mode == 'validation':
            files = glob.glob(
                os.path.join(
                    self.data_path, root_folder, self.machine_type, 'test',
                    '*id_{:02d}_*.wav'.format(self.machine_id)
                )
            )
        else:
            raise AttributeError
        assert len(files) > 0
        return sorted(files)


if __name__ == '__main__':

    for type_, id_ in enumerate_development_datasets():
        _ = MachineDataSet(type_, id_, data_path=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'), mode='training')
        _ = MachineDataSet(type_, id_, data_path=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'), mode='validation')

    for type_, id_ in enumerate_evaluation_datasets():
        _ = MachineDataSet(type_, id_, data_path=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'), mode='training')
        _ = MachineDataSet(type_, id_, data_path=os.path.join(os.path.expanduser('~'), 'shared', 'dcase2020_task2'), mode='validation')
