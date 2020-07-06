from dcase2020_workshop.data_sets import INVERSE_CLASS_MAP, TRAINING_ID_MAP
import numpy as np
from scipy.stats import rankdata

def get_experiment_by_name(name, runs, filters={}):
    experiment_dict = dict()
    for i in range(6):
        experiment_dict[i] = dict()

    experiment_dict['name'] = name

    for experiment in runs:
        flag = True
        for k in filters:
            parts = k.split('.')

            tmp = experiment['config']

            for p in parts:
                if p not in tmp:
                    flag = False
                else:
                    tmp = tmp.get(p)

            if tmp != filters[k]:
                flag=False

        if experiment['config'].get('id') == name and flag:
            machine_dict = experiment_dict.get(experiment['config']['machine_type'])
            result = experiment.get('result')
            machine_type = INVERSE_CLASS_MAP[experiment['config']['machine_type']]
            machine_id = experiment['config']['machine_id']
            if result:
                auc = result[machine_type][ f'json://{machine_id}']['py/tuple'][0]
                pauc = result[machine_type][ f'json://{machine_id}']['py/tuple'][1]
                machine_dict.setdefault(machine_id, []).append([auc, pauc])
            #else:

                #machine_dict.setdefault(machine_id, []).append([0, 0])

    for i in range(6):
        for j in TRAINING_ID_MAP[i]:
            if experiment_dict[i].get(j) is None:
                print(f'{i} {j} experiment not finished.')
                experiment_dict[i][j] = [[0, 0]]

    return experiment_dict


def to_flat_record(experiment):
    record = []
    for i in range(6):
        for j in TRAINING_ID_MAP[i]:
                if any(experiment.get(i)) and any(experiment.get(i).get(j)):
                    record.append(experiment[i][j])
                else:
                    record.append([0, 0])
    assert len(record) == 23
    return  experiment['name'], record


def rank_array(auc_means, pauc_means):
    auc_ranks = []
    pauc_ranks = []
    idxes = [0, 4, 8, 12, 16, 19, 23]
    best_idxes = []
    for type_, (i, j) in enumerate(zip(idxes[:-1], idxes[1:])):
        average_auc = auc_means[:, i:j].mean(axis=1)
        average_pauc = pauc_means[:, i:j].mean(axis=1)
        best_idxes.append(
            np.argsort(average_auc + average_pauc)[::-1]
        )
        print(f'Best Model for Machine Type {type_}: {best_idxes[-1]}')
        auc_ranks.append(rankdata(-average_auc))
        pauc_ranks.append(rankdata(-average_pauc))

    ranks = np.stack([np.array(list(zip(*auc_ranks))), np.array(list(zip(*pauc_ranks)))], axis=-1).mean(axis=-1).mean(
        axis=-1)

    return list(np.argsort(ranks)), best_idxes