import h5py
import getpass
import os
import csv
from .data_for_TA import POST_ID_STRING, BASE_READ_PATH, BASE_WRITE_PATH, GROUP_FILE_PATH, GROUP_PATH, SHARED_PATH, NAME_CONV, TEST_NAME_POST_FIX, BIN_SIZE, POST_STIM_WINDOW_SIZE, N_PARAMS
import numpy as np
import math

def get_group_id(user_id):
    with open(os.path.join(BASE_READ_PATH, GROUP_FILE_PATH), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            group_id, *members = [int(x.strip()) for x in row]
            if user_id in members:
                return group_id
    raise ValueError(f"No group found for user {user_id}")  # user has no friends

def load_data(
    network: int, 
    day_in_vitro: int, 
    use_group_data: bool = False, 
    test_mode: bool = False
    ):

    username = getpass.getuser()
    user_id  = int(username.split("_")[1].split(POST_ID_STRING)[0])
    group_id = get_group_id(user_id)
    directory = GROUP_PATH(group_id) if use_group_data else SHARED_PATH
    post_fix = TEST_NAME_POST_FIX + ".h5" if test_mode else ".h5"
    data_path = os.path.join(BASE_READ_PATH,directory,NAME_CONV(network,day_in_vitro))
    data_path = data_path + post_fix
    with h5py.File(data_path, 'r') as h5file:
        stimulation_parameters = h5file.get("stimulation_parameters", None)
        stimulation_parameters = None if stimulation_parameters is None else stimulation_parameters[()]
        if stimulation_parameters is not None:
            stimulation_patterns = stimulation_parameters[...,1].copy()
        else:
            stimulation_patterns = None
        binned_spike_train_responses = h5file.get("binned_spike_train_responses", None)
        binned_spike_train_responses = None if binned_spike_train_responses is None else binned_spike_train_responses[()]
        stimulation_times = h5file["stimulation_times"][()]
        electrodes = h5file["electrodes"][()]
        impedance_map = h5file["impedance_map"][()]
    return stimulation_parameters, stimulation_patterns, binned_spike_train_responses, stimulation_times, impedance_map, electrodes

def save_data(
    network: int, 
    day_in_vitro: int, 
    binned_spike_train_responses: np.ndarray = None, 
    stimulation_parameters: np.ndarray = None, 
    stimulation_patterns: np.ndarray = None, 
    use_group_data: bool = False, 
    test_mode: bool = False
    ):

    username = getpass.getuser()
    user_id  = int(username.split("_")[1].split(POST_ID_STRING)[0])
    group_id = get_group_id(user_id)
    directory = GROUP_PATH(group_id) if use_group_data else os.path.join(SHARED_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)
    post_fix = TEST_NAME_POST_FIX if test_mode else ""
    data_path = os.path.join(BASE_READ_PATH,directory,NAME_CONV(network,day_in_vitro))
    data_path = data_path + post_fix + ".h5"
    # Checking shapes
    with h5py.File(data_path, 'r') as h5file:
        if "binned_spike_train_responses" in h5file:
            size = h5file["binned_spike_train_responses"].shape[0]
        else:
            size = h5file["stimulation_parameters"].shape[0]
        n_channel = h5file["electrodes"].shape[0]
    if stimulation_parameters is not None:
        assert (size,N_PARAMS) == stimulation_parameters.shape, f"Shape mismatch between gt {(size,N_PARAMS)} and input {stimulation_parameters.shape}"
    if stimulation_patterns is not None:
        assert (size,) == stimulation_patterns.shape, f"Shape mismatch between gt {(size,)} and input {stimulation_patterns.shape}"
    if binned_spike_train_responses is not None:
        assert (size,n_channel,math.ceil(POST_STIM_WINDOW_SIZE/BIN_SIZE)) == binned_spike_train_responses.shape, f"Shape mismatch between gt {(size,n_channel,POST_STIM_WINDOW_SIZE//BIN_SIZE)} and input {stimulation_parameters.shape}"

    versions = range(1,4)
    directory = GROUP_PATH(group_id) if use_group_data else os.path.join(SHARED_PATH,GROUP_PATH(group_id))
    for v in versions:
        save_path = os.path.join(BASE_WRITE_PATH,directory,NAME_CONV(network,day_in_vitro)+post_fix+f"_v{v}.h5")
        if not os.path.isfile(save_path):
            break
        if v == versions[-1]:
            print(f"Found {len(versions)} saves for this dataset. Deleting the oldest one.")
            os.remove(os.path.join(BASE_WRITE_PATH, directory, NAME_CONV(network,day_in_vitro)+post_fix+f"_v{versions[0]}.h5"))
            for old_v in versions[1:]:
                old = os.path.join(BASE_WRITE_PATH, directory, NAME_CONV(network,day_in_vitro)+post_fix+f"_v{old_v}.h5")
                new_v = versions[versions.index(old_v) - 1]
                new = os.path.join(BASE_WRITE_PATH, directory, NAME_CONV(network,day_in_vitro)+post_fix+f"_v{new_v}.h5")
                os.rename(old, new)            

    with h5py.File(save_path, 'w') as h5file:
        if binned_spike_train_responses is not None:
            h5file.create_dataset("binned_spike_train_responses", data=binned_spike_train_responses, compression=1)
        if stimulation_parameters is not None:
            h5file.create_dataset("stimulation_parameters", data=stimulation_parameters, compression=1)
        if stimulation_patterns is not None:
            h5file.create_dataset("stimulation_patterns", data=stimulation_patterns, compression=1)
    print("Saved the data!")
 