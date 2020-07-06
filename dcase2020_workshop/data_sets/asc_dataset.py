import os
import torch.utils.data
import glob
from dcase2020_workshop.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, EVALUATION_ID_MAP, ALL_ID_MAP,\
    enumerate_development_datasets, enumerate_evaluation_datasets
import librosa
import numpy as np
from dcase2020_workshop.data_sets import MCMDataSet
from dcase2020_workshop.data_sets.audio_processor import __load_preprocess_file__
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class ASCSet(torch.utils.data.Dataset, BaseDataSet):

    def __init__(
            self,
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

            **kwargs
    ):
        self.data_path = os.path.join(data_path, 'dcase2020_task1', 'audio')

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

    @property
    def observation_shape(self) -> tuple:
        return 1, self.audio_features['num_mel'], self.window_length

    def training_data_set(self):
        return self

    def validation_data_set(self):
        return self

    def __getitem__(self, item):
        file_idx, offset = self.index_map[item]
        observation = self.data[file_idx][:, offset:offset + self.window_length]
        meta_data = self.meta_data[file_idx].copy()
        meta_data['observations'] = observation[None]
        return meta_data

    def __len__(self):
        return self.length

    def __get_file_list__(self):
        files = glob.glob(os.path.join(self.data_path, '*.wav'))
        assert len(files) > 0
        return sorted(files)

    def __load_data__(self):

        file_id = '{sr}_{num_mel}_{n_fft}_{hop_size}_{power}_{fmin}_{fmax}_{normalize_raw}_{normalize_spec}'.format(
            **self.audio_features
        )
        data_path = os.path.join(self.data_path, file_id + ".npz")
        meat_data_path = os.path.join(self.data_path, file_id + ".pkl")


        data = []
        meta_data = []
        if os.path.exists(data_path) and os.path.exists(meat_data_path):
            print(f'Loading ASC data ...')
            container = np.load(data_path)
            data = [container[key] for key in container]
            with open(meat_data_path, "rb") as f:
                meta_data = pickle.load(f)
        else:
            print(f'Loading & Saving ASC data ...')
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
        return {
            'targets': 1,
            'machine_types': -1,
            'machine_ids': -1,
            'file_ids': os.sep.join(os.path.normpath(file_path).split(os.sep)[-4:])
        }


if __name__ == '__main__':
    from dcase2020_workshop.data_sets import DATA_PATH
    a = ASCSet(data_path=DATA_PATH).training_data_set()[0]

    print(a)



