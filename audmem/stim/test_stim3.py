# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:07:59 2018

@author: Thomas Schatz & Marianne Duyck

Repeated white noise stimuli creation

For a second ERP test.

Create 16-bit (signed) 44.1kHz no header mono wavefiles
same design as Andrillon et al. 2015 for one block
3 trial types: RefRN, RN and noise, separated by 2-3s of silence
For RefRN and RN, stimuli are 0.2s separated by 0.3s noise and 5 repetitions
Outputs:
  1. the noise for one block (5.4min)
  2. the indicator that is a state-change indicator of trial/silence onsets
  3. a tab delimited text file listing the orders of conditions.

Usage:
  python test_stim3.py path/2/output/folder random_seed
"""

import numpy as np
import soundfile
import random
import datetime, os
import pandas as pd
import argparse
import json


def get_n_samples(dur_s, fs):
    return int(np.floor(fs*dur_s))


def make_stim_trial(array_stim, ISI_dur, n_reps, fs):
    list_trial = []
    n_sampISI = get_n_samples(ISI_dur, fs)
    for i in range(n_reps):
        list_trial.append(np.random.randn(n_sampISI))
        list_trial.append(array_stim)
    return np.concatenate(list_trial)


def make_noise_trial(trial_dur, fs):
    n_samp = get_n_samples(trial_dur, fs)
    return np.random.randn(n_samp)


def mydatetime(dt):
    return dt.strftime('%Y-%m-%d-%H-%M-%S')


def create_all_stims(random_seed, fs=44100, stim_dur=.2,
                     ISI_dur=.3, ITI_min=2., ITI_max=3., 
                     n_reps=5, block=None):
    """
    Params:
      fs: sampling frequency in Hz

      stim_dur: repeated noise duration (s)
      ISI_dur: duration of noise between two repetitions of repeated noise (s)
      ITI_min, ITI_max: range of duration of silence between two trials (s)
        (uniformly distributed in this range)
      n_reps: number of noise repetitions within trial
      block: dict indicating the number of blocks of each type desired
    """
    if block is None:
        block = {"RefRN": 16, "RN": 16, "noise": 32}
    trial_dur = n_reps*(ISI_dur+stim_dur)

    # prepare orders of trials in block
    cond_list = []
    for trial_type in block.keys():
        cond_list+=[trial_type]*block[trial_type]
    random.shuffle(cond_list)

    # set randomseed
    np.random.seed(random_seed)

    # get number of sample for repeated noise template
    n_sampStim = get_n_samples(stim_dur, fs)

    # generate RefRN template
    REF = np.random.randn(n_sampStim)

    # generate one block
    trials = []
    for trial_type in cond_list:
        if trial_type == "RefRN":
            trials.append(make_stim_trial(REF, ISI_dur, n_reps, fs))
        elif trial_type == "RN":
            RN = np.random.randn(n_sampStim)
            trials.append(make_stim_trial(RN, ISI_dur, n_reps, fs))
        else:
            trials.append(make_noise_trial(trial_dur, fs))

    # adds silence between trials and end
    for t in range(len(trials)):
        ITI_array = np.zeros(shape=(get_n_samples(random.uniform(ITI_min, ITI_max), fs)), dtype=np.float)
        trials.insert(2*t+1, ITI_array)
        cond_list.insert(2*t+1, "silence")
    # adds silence beginning block
    ITI_array = np.zeros(shape=(get_n_samples(random.uniform(ITI_min, ITI_max), fs)), dtype=np.float)
    trials.insert(0, ITI_array)
    cond_list.insert(0, "silence")

    df_design = pd.DataFrame()
    df_design["cond"] = cond_list
    df_design["dur"] = [len(trials[i])/float(fs) for i in range(len(trials))]

    # trial indicator: at 0 during trial
    indicators = [[1.]*len(trial) if i % 2 == 0 else [0.]*len(trial)
                for i, trial in enumerate(trials)]

    noise = np.concatenate(trials)
    indicator = np.concatenate(indicators)

    assert len(indicator) == len(noise)

    # signed int 16 format
    nb_levels = 65536
    mini = -32768
    maxi = 32767

    # scale noise to occupy range while
    # keeping clipping probability low
    K = maxi/10.
    noise = K*noise
    indicator = K*indicator

    # quantize
    noise = np.round(noise)
    indicator = np.round(indicator)

    # check there is no clipping
    assert np.max(noise) <= maxi
    assert np.min(noise) >= mini

    # convert to signed 16 bit integer
    noise = noise.astype(np.int16)
    indicator = indicator.astype(np.int16)

    params = {'random_seed': random_seed, 'fs': fs, 
              'stim_dur': stim_dur, 'ISI_dur': ISI_dur,
              'ITI_min': ITI_min, 'ITI_max': ITI_max, 
              'n_reps': n_reps, 'block': block}
    return df_design, noise, indicator, params


def save_sound(sound_array, path):
    soundfile.write(path,
                    sound_array,
                    samplerate=44100,
                    subtype='PCM_16')


def save_design(df, path):
    df.to_csv(path, sep='\t', index=False)


def save_params(params, path):
  with open(path, 'w') as out_file:
    #yaml.dump(params, out_file, default_flow_style=False)
    json.dump(params, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', metavar='path',
                        help='the path to the folder for all outputs')
    parser.add_argument('random_seed', type=int,
                        help='random generator seed')
    args = parser.parse_args()
    # print(args)
    df_design, noise, indicator, params = create_all_stims(args.random_seed)
    dt = mydatetime(datetime.datetime.now())
    fileroot = os.path.join(args.output_folder, dt+'_seed'+str(args.random_seed))

    save_design(df_design, fileroot+'_design.txt')
    save_sound(noise, fileroot+'_noise.wav')
    save_sound(indicator, fileroot+'_indicator.wav')
    save_params(params, fileroot+'_params.txt')
