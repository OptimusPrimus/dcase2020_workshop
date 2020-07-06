import os
import torch.utils.data
from dcase2020_workshop.data_sets import BaseDataSet, CLASS_MAP, INVERSE_CLASS_MAP, TRAINING_ID_MAP, ALL_ID_MAP
from dcase2020_workshop.data_sets import MachineDataSet
import copy
import numpy as np

VALID_TYPES = {
    'different_mic': {
        0: [3, 4],
        1: [3, 4],
        2: [3, 4],
        5: [3, 4],
        3: [0, 1, 2, 5],
        4: [0, 1, 2, 5],
    },
    'same_mic_same_type': {
        0: [0],
        1: [1],
        2: [2],
        5: [5],
        3: [3],
        4: [4],
    },
    'same_mic_different_type': {
        0: [1, 2, 5],
        1: [0, 2, 5],
        2: [0, 1, 5],
        5: [0, 1, 2],
        3: [4],
        4: [3],
    },
    'same_mic_all_types': {
        0: [0, 1, 2, 5],
        1: [1, 0, 2, 5],
        2: [2, 0, 1, 5],
        5: [5, 0, 1, 2],
        3: [3, 4],
        4: [4, 3],
    },
    'all': {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 1, 2, 3, 4, 5],
        2: [0, 1, 2, 3, 4, 5],
        5: [0, 1, 2, 3, 4, 5],
        3: [0, 1, 2, 3, 4, 5],
        4: [0, 1, 2, 3, 4, 5],
    }
}


class ComplementMCMDataSet(BaseDataSet):

    def __init__(
            self,
            num_samples=None,
            num_classes=None,
            blur=None,
            **kwargs
    ):

        assert type(kwargs['machine_type']) == int and type(kwargs['machine_id']) == int
        assert kwargs['machine_id'] >= 0
        assert kwargs['machine_type'] >= 0

        kwargs['data_path'] = os.path.join(kwargs['data_path'], 'dcase2020_task2')
        kwargs['blur'] = blur

        self.kwargs = kwargs
        self.valid_types = kwargs['valid_types']
        del kwargs['valid_types']

        training_sets = []

        lenghts = []
        for type_ in VALID_TYPES[self.valid_types][kwargs['machine_type']]:
            for id_ in ALL_ID_MAP[type_]:
                if type_ != kwargs['machine_type'] or id_ != kwargs['machine_id']:
                    kwargs_ = copy.deepcopy(kwargs)
                    kwargs_['machine_type'] = type_
                    kwargs_['machine_id'] = id_
                    t = MachineDataSet(mode='training', **kwargs_)
                    lenghts.append(len(t))
                    training_sets.append(t)

        self.training_set = torch.utils.data.ConcatDataset(training_sets)

        if num_samples is not None:
            assert num_samples > 0
            num_samples = min(num_samples, len(self.training_set))

            indices = np.random.choice(len(self.training_set), size=num_samples, replace=False)

            # TODO: replace this hacky solution
            if num_samples < 64:
                indices = np.random.choice(indices, size=64, replace=True)

            self.training_set = torch.utils.data.Subset(self.training_set, indices)


    @property
    def observation_shape(self) -> tuple:
        return 1, self.kwargs['num_mel'], self.kwargs['window_length']

    def training_data_set(self):
        return self.training_set

    def validation_data_set(self):
        raise NotImplementedError

