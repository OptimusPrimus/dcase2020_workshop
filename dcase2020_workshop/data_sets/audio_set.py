import torch.utils.data
import torch.utils.data
import torch.utils.data
import glob
from dcase2020_workshop.data_sets import BaseDataSet
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dcase2020_workshop.data_sets.audio_processor import __load_preprocess_file__
from tqdm import tqdm
import pickle
import os


class AudioSet(BaseDataSet):

    def __init__(
            self,
            **kwargs
    ):

        kwargs['data_path'] = os.path.join(kwargs['data_path'], 'audioset', 'audiosetdata')

        self.kwargs = kwargs

        class_names = sorted(
            [
                class_name for class_name in os.listdir(kwargs['data_path']) if os.path.isdir(os.path.join( kwargs['data_path'], class_name))
            ]
        )

        if len(class_names) == 0:
            class_files = glob.glob(os.path.join(kwargs['data_path'], '*.npz'))
            # TODO: change this hacky solution
            class_names = sorted([os.path.split(class_file)[-1].split('_None_')[-1][:-4]for class_file in class_files])

        training_sets = []
        for class_name in class_names:
            training_sets.append(AudioSetClassSubset(class_name, **kwargs))

        self.training_set = torch.utils.data.ConcatDataset(training_sets)
        self.validation_set = None

    @property
    def observation_shape(self) -> tuple:
        return 1, self.kwargs['num_mel'], self.kwargs['window_length']

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        return self.validation_set


class AudioSetClassSubset(torch.utils.data.Dataset):

    def __init__(
            self,
            class_name,
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

            window_length=256,
            window_hop_size=None,

            max_files_per_class=-1,

            **kwargs
    ):
        self.class_name = class_name
        self.data_path = data_path
        self.max_files_per_class = max_files_per_class

        self.audio_features = {
            'sr': sr,
            'num_mel': num_mel,
            'n_fft': n_fft,
            'hop_size': hop_size,
            'power': power,
            'fmin': fmin,
            'fmax': fmax,
            'normalize_raw': normalize_raw,
            'normalize_spec': normalize_spec
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
        observation = self.data[file_idx][:, offset:offset + self.window_length]
        meta_data = self.meta_data[file_idx].copy()
        meta_data['observations'] = observation[None]
        return meta_data

    def __len__(self):
        return self.length

    def __get_file_list__(self):
        files = glob.glob(os.path.join(self.data_path, self.class_name, '*.wav'))
        assert len(files) > 0
        return sorted(files)

    def __load_data__(self):

        file_id = '{sr}_{num_mel}_{n_fft}_{hop_size}_{power}_{fmin}_{fmax}_{normalize_raw}_{normalize_spec}_{class_name}'.format(
            class_name=self.class_name,
            **self.audio_features
        )
        data_path = os.path.join(self.data_path, file_id + ".npz")
        meat_data_path = os.path.join(self.data_path, file_id + ".pkl")

        data = []
        meta_data = []

        if os.path.exists(data_path) and os.path.exists(meat_data_path):
            print(f'Loading data set for {self.class_name}..')
            container = np.load(data_path)
            data = [container[key] for key in container]
            with open(meat_data_path, "rb") as f:
                meta_data = pickle.load(f)
        else:
            print(f'Loading & Saving data set for {self.class_name}...')
            files = self.__get_file_list__()
            def closure(f):
                return __load_preprocess_file__(
                    f,
                    config=self.audio_features,
                    trim_zeros=True,
                    min_size=self.window_length+50,
                    max_duration=15
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                for f, r in tqdm(zip(files, executor.map(closure, files)), total=len(files)):
                    if r is not None:
                        data.append(r)
                        meta_data.append(self.__get_meta_data__(f))

            if self.max_files_per_class != -1 and len(data) > self.max_files_per_class:
                data = data[:self.max_files_per_class]
                meta_data = meta_data[:self.max_files_per_class]

            np.savez(data_path, *data)
            with open(meat_data_path, "wb") as f:
                pickle.dump(meta_data, f)
        return data, meta_data

    def __get_meta_data__(self, file_path):
        return {
            'targets': 1,
            'machine_types': -1,
            'machine_ids': -1,
            'file_ids': os.sep.join(os.path.normpath(file_path).split(os.sep)[-4:])
        }


if __name__ == '__main__':
    a = audio_set = AudioSet(data_path=os.path.join(os.path.expanduser('~'), 'shared')).training_data_set()[0]

    print(a)



