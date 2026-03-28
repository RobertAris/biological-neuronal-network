import numpy as np
import h5py
import json
import argparse
import os
import math

"""
For TAs:

Each year you will have to create these directories with the course master account. 
To minimize any accidental overwrite and error traps, the global variables enable simple directory finding for the students. 
We provide them with utility functions for loading and saving. 

1. data_sets
In data_sets you can put any data that is readable by the students, but not writable. 
This means, students cannot accidentally delete any data in this directory. Do this with:
chmod -R -c 0750
for the directory. 
Currently it is built such that each group has one folder and there is an additional shared folder. 

2. saves
Here you collect results, saves from students. Here you have to give also writing access:
chmod -R -c 0770
Currently it is built such that each group has one folder in saves, and one in saves/shared/

3. groups.txt
This contains the assignement for a group to the user ids. The first entry is the group. For example:
group_id, members
1, 1, 2
2, 3, 4
3, 14, 5
4, 6, 7
5, 8, 9
6, 10, 11
7, 12, 13
8, 15, 16

That way, their user account is automatically assigned to the group and no accidental switch up can happen. 

This script creates a dataset with the parameters specified below from a pattern_sweep recording. 
"""

POST_ID_STRING          = "fs"
BASE_READ_PATH          = f"/usr/itetnas04/data-scratch-02/bnn_fs26/data/data_sets"
BASE_WRITE_PATH         = f"/usr/itetnas04/data-scratch-02/bnn_fs26/data/saves"
GROUP_FILE_PATH         = "groups.txt"
GROUP_PATH              = lambda group: f"g{int(group)}"
SHARED_PATH             = "shared/"
NAME_CONV               = lambda net, div: f"N{int(net)}_DIV{int(div)}"
TEST_NAME_POST_FIX      = "_test"
BIN_SIZE                = 5
POST_STIM_WINDOW_SIZE   = 400
N_PARAMS                = 2

def getResponse(spike_times: np.ndarray, starts: np.ndarray, spike_matrix: np.ndarray = None,
                starts_matrix: np.ndarray = None, window: (int,int) = (0, 200)
                ) -> (np.ndarray,np.ndarray):
    """
    Based on an array containing the start timings of a response, the spikes are labeled with the corresponding response.
    :param spike_times: A 1d numpy array containing the spike timings (in frameno).
    :param starts: A 1d numpy array containing the trigger times.
    :param spike_matrix: A 2d numpy array of shape (N, n_spikes) containing addtional information per spike.
    :param starts_matrix: A 2d numpy array of shape (N, n_spikes) containing addtional information per trigger time.
    :param window: Window, in which a spike has to occur such that it belongs to a specific response start.
    :return: A numpy array of shape (5,n_filtered_spikes), where the first dimension corresponds to
    (channel, additional_info, response) and the used response starts (trigger_time, additional_info).
    """
    if len(starts)<=0:
        print("Warning: Array with start timings is empty")
    elif np.min(starts) < -window[0] or window[1] <= window[0]:
        raise Exception("Inappropriate window.")

    order = np.argsort(starts)
    sorted_starts = starts[order]
    keep = np.ones(sorted_starts.size, dtype=bool)
    while True:
        current = sorted_starts[keep]
        if current.size <= 1:
            break
        diffs = np.diff(current)
        too_close = np.where(diffs < window[1]-window[0])[0]
        if too_close.size == 0:
            break
        keep_idx = np.flatnonzero(keep)
        cluster_starts = np.r_[True, np.diff(too_close) > 1]
        cluster_first = too_close[cluster_starts]
        drop_pos_in_current = cluster_first + 1
        drop_idx_in_sorted = keep_idx[drop_pos_in_current]
        keep[drop_idx_in_sorted] = False
    kept_sorted_idx = np.flatnonzero(keep)
    new_starts = starts[np.sort(order[kept_sorted_idx])].ravel()
    if len(new_starts) != len(starts):
        print(f"{len(starts)-len(new_starts) } starts that were too close were dropped.")

    # keep original order information
    orig_idx = np.arange(new_starts.size)
    sort_idx = np.argsort(new_starts)
    sorted_starts = new_starts[sort_idx]

    # find the last start before each spike in the *sorted* array
    idx_sorted = np.searchsorted(sorted_starts, spike_times, side="right") - 1

    valid = idx_sorted >= 0
    delay = spike_times - sorted_starts[idx_sorted]
    valid &= (delay >= window[0]) & (delay < window[1])

    n_features_spike = 2 + (0 if spike_matrix is None else spike_matrix.shape[0])
    n_features_start = 1 + (0 if starts_matrix is None else starts_matrix.shape[0])
    responses = np.zeros((n_features_spike,np.count_nonzero(valid)))
    response_starts = np.zeros((n_features_start, len(new_starts)))
    # filter spikes
    responses[0] = delay[valid]
    responses[-1] = orig_idx[sort_idx[idx_sorted[valid]]]
    if spike_matrix is not None:
        responses[1:-1] = spike_matrix[:, valid]
    response_starts[0] = new_starts
    if starts_matrix is not None:
        response_starts[1:] = starts_matrix[:, order[kept_sorted_idx]]
    return responses, response_starts

def create_binned_spike_train(*spike_times, trace_size, bin_size):
    """
    Converts spike times to a binned spike trace.
    :param spike_times: tuple of spike indices, where the last is the one to be binned.
    The first dimensions represent indices of the output trace for the other dimensions. 
    :param trace_size: Dimensions of the output, where the last dimension is to be binned.
    :param bin_size: Bin size for the last dimension
    """
    n_bins = math.ceil(trace_size[-1]/bin_size)
    binned_trace = np.zeros((*trace_size[:-1],n_bins))
    np.add.at(binned_trace,(*spike_times[:-1], spike_times[-1] // bin_size),1)
    return binned_trace

def load_data(data_path):
    blank_time_in_samples = 30
    with h5py.File(data_path, 'r') as h5file:
        spike_times = h5file["spikes"]["frameno"] # Time in ms
        spike_channels = h5file["spikes"]["channel"]
        spike_amplitudes = h5file["spikes"]["amplitude"]
        spike_electrode = h5file["spikes"]["electrode"]
        event_times = h5file["events"]["frameno"]
        event_ids = h5file["events"]["eventid"]
        event_dict = h5file["events"]["eventMessage"]
        # This one had only different amplitudes, the amplitude was saved in the dictionary under “amp“
        stim_amplitudes = np.array([float(json.loads(m.decode())['amp']) for m in event_dict])
        stim_frequencies = np.array([float(json.loads(m.decode())['freq']) for m in event_dict])
        stim_patterns = np.array([float(json.loads(m.decode())['pattern']) for m in event_dict])
        stim_times = event_times
        electrode_channel_mapping = h5file["Meta_Info"]["channel_mapping"][:]

    responses, starts = getResponse(
        spike_times,
        event_times,
        np.concatenate((spike_channels[None],spike_amplitudes[None]),axis=0),
        np.concatenate((stim_frequencies[None],stim_patterns[None],stim_times[None]),axis=0),
        window=(0,POST_STIM_WINDOW_SIZE)
    )
    responses = responses[:,responses[0]>blank_time_in_samples]
    value_to_index = {v: i for i, v in enumerate(electrode_channel_mapping[1])}
    channel_indices = np.vectorize(value_to_index.get)(responses[1])
    binned_spike_train = create_binned_spike_train(
        responses[-1].astype(int),channel_indices.astype(int),responses[0].astype(int),
        trace_size=[len(event_times),electrode_channel_mapping.shape[-1],POST_STIM_WINDOW_SIZE],
        bin_size=BIN_SIZE
    ).clip(0,1)

    parameters = np.concatenate((stim_frequencies[:,None], stim_patterns[:,None]),axis=-1)

    return parameters, binned_spike_train, stim_times[:,None], electrode_channel_mapping[0]

def save_dataset(
    data_path, 
    voltage_map_path, 
    network: int, 
    day_in_vitro: int, 
    group: int = None, 
    param_only = False, 
    response_only = False,
    base_path = "."
    ):
    if param_only and response_only:
        raise ValueError("For partial dataset creation only one of {Stimulation Parameters, Binned Spike Train Responses} can be specified.")
    
    directory = os.path.join(base_path, GROUP_PATH(group_id) if group is not None else SHARED_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)
    post_fix = TEST_NAME_POST_FIX if param_only or response_only else ""
    save_path = os.path.join(directory,NAME_CONV(network,day_in_vitro))
    save_path = save_path + post_fix + ".h5"

    stimulation_parameters, binned_spike_train_responses, stimulation_times, electrodes = load_data(data_path)
    impedance_map = np.load(voltage_map_path)
    with h5py.File(save_path, "w") as h5file:
        if response_only:
            h5file.create_dataset("binned_spike_train_responses", data=binned_spike_train_responses, compression=1, dtype="float32")
        elif param_only:
            h5file.create_dataset("stimulation_parameters", data=stimulation_parameters, compression=1, dtype="float32")
        else:
            h5file.create_dataset("binned_spike_train_responses", data=binned_spike_train_responses, compression=1, dtype="float32")
            h5file.create_dataset("stimulation_parameters", data=stimulation_parameters, compression=1, dtype="float32")
        h5file.create_dataset("stimulation_times", data=stimulation_times, compression=1)
        h5file.create_dataset("electrodes", data=electrodes)
        h5file.create_dataset("impedance_map", data=impedance_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spike detection script."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input file containing the processed neural data."
    )
    parser.add_argument(
        "voltage_map_path",
        type=str,
        help="Path to the impedance map."
    )
    parser.add_argument(
        "network",
        type=int,
        help="What network is being saved"
    )
    parser.add_argument(
        "day_in_vitro",
        type=int,
        help="DIV of the network at the time of recording."
    )
    parser.add_argument(
        "--group",
        type=int,   # or int, or str, whatever you need
        default=None,
        help="What group the recording belongs to. It is saved in a shared directory if not specified."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=".",
        help="Where the basepath is of the data."
    )
    parser.add_argument(
        "--param_only",
        action="store_true",
        help="If set, only parameters are saved. This can be used as testing data without providing the ground truth."
    )
    parser.add_argument(
        "--response_only",
        action="store_true",
        help="If set, only responses are saved. This can be used as testing data without providing the ground truth."
    )

    args = parser.parse_args()
    save_dataset(args.data_path, args.voltage_map_path, args.network, args.day_in_vitro, args.group, args.param_only, args.response_only, args.base_path)