import librosa
import numpy as np


def __load_preprocess_file__(file, config=None, trim_zeros=False, min_size=None, max_duration=None):
    try:
        x, sr = librosa.load(file, sr=config['sr'], mono=True, duration=max_duration)
    except ValueError:
        print(f'{file} not formatted correctly.')
        return None

    if trim_zeros:
        x = np.trim_zeros(x)

    if np.all(x == 0) or len(x) < config['n_fft']:
        print(f'{file} is empty.')
        return None

    if config['normalize_raw'] == 'unit_variance':
        # TODO: - mean() ?
        x = x / x.std()
    else:
        raise AttributeError

    x = librosa.feature.melspectrogram(
        y=x,
        sr=config['sr'],
        n_fft=config['n_fft'],
        hop_length=config['hop_size'],
        n_mels=config['num_mel'],
        power=config['power'],
        fmin=config['fmin'],
        fmax=config['fmax']
    )

    if min_size is not None and x.shape[-1] < min_size:
        print(f'{file} is too short.')
        return None

    if config['power'] == 1:
        x = librosa.core.amplitude_to_db(x)
    elif config['power'] == 2:
        x = librosa.core.power_to_db(x)
    else:
        raise AttributeError

    if config['normalize_spec'] == 'zero_mean_unit_variance':
        x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)

    return x
