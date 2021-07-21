from collections import OrderedDict

import h5py
import numpy as np
from scipy.sparse import spdiags


def read_txt(filelists, data_dir):
    all_path = []
    for file_path in filelists:
        with open(file_path, 'r') as f:
            test_paths = f.read().split('\n')
            test_paths = [data_dir + '/' + path for path in test_paths]
            all_path.append(test_paths)
    return all_path


def _read_list(file_path, data_dir):
    with open(file_path, 'r') as f:
        test_paths = f.read().split('\n')
        test_paths = [data_dir + path for path in test_paths]
    return test_paths


def read_from_single_txt(train_path, data_dir):
    all_train_paths = _read_list(train_path, data_dir)
    return all_train_paths


def read_from_txt(train_path, test_path, data_dir):
    all_train_paths = _read_list(train_path, data_dir)
    all_test_paths = _read_list(test_path, data_dir)
    return all_train_paths, all_test_paths


def get_nframe_video(path, dataset):
    temp_f1 = h5py.File(path, 'r')  # This might be changed
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]  # It was 36000, but in this dataset it should be 8999
    return nframe_per_video


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def mag2db(mag):
    """Convert a magnitude to decibels (dB)
    If A is magnitude,
        db = 20 * log10(A)
    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes
    Returns
    -------
    db : float or ndarray
        corresponding values in decibels
    """
    return 20. * np.log10(mag)
