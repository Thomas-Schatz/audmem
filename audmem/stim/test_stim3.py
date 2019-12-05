# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:07:59 2018

@author: Thomas Schatz & Marianne Duyck

Repeated white noise stimuli creation

For a second ERP test.
"""

import numpy as np
import soundfile
import random
import datetime, os
import pandas as pd
import argparse

# Create 16-bit (signed) 44.1kHz no header mono wavefiles
# same design as Andrillon et al. 2015 for one block
# 3 trial types: RefRN, RN and noise, separated by 2-3s of silence
# For RefRN and RN, stimuli are 0.2s separated by 0.3s noise and 5 repetitions
# outputs:
# 1. the noise for one block (5.4min)
# 2. the indicator that is a state-change indicator of trial/silence onsets
# 3. a tab delimited text file listing the orders of conditions.

# params
fs = 44100  # Hertz
Stim_dur = 0.2  # seconds
ISI_dur = 0.3  # seconds
ITI_min, ITI_max = 2., 3.  # seconds
trial_dur = 2.5 #seconds
n_reps = 5 # reps of noise within trials
block = {"RefRN": 16, "RN": 16, "noise": 32}

# functions
def get_n_samples(dur_s, fs):
    return int(np.floor(fs*dur_s))


def add_silence(list_stim, dur_s, fs):
    n_samp = get_n_samples(fs, dur_s)
    list_stim.append([0.5]*n_samp)


def make_stim_trial(array_stim, ISI_dur):
    list_trial = []
    n_sampISI = get_n_samples(ISI_dur, fs)
    for i in range(n_reps):
        list_trial.append(np.random.randn(n_sampISI))
        list_trial.append(array_stim)
    return np.concatenate(list_trial)


def make_noise_trial(trial_dur):
    n_samp = get_n_samples(trial_dur, fs)
    return np.random.randn(n_samp)


def mydatetime(dt):
    return dt.strftime('%Y-%m-%d-%H-%M-%S')


def create_all_stims(randomseed):
    # prepare orders of trials in block
    cond_list = []
    for trial_type in block.keys():
        cond_list+=[trial_type]*block[trial_type]
    random.shuffle(cond_list)

    # set seed and generate RefRN
    np.random.seed(randomseed)
    n_sampStim = get_n_samples(Stim_dur, fs)
    REF = np.random.randn(n_sampStim)

    # generate one block
    trials = []
    for trial_type in cond_list:
        if trial_type == "RefRN":
            trials.append(make_stim_trial(REF, ISI_dur))
        elif trial_type == "RN":
            RN = np.random.randn(n_sampStim)
            trials.append(make_stim_trial(RN, ISI_dur))
        else:
            trials.append(make_noise_trial(trial_dur))

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

    return df_design, noise, indicator

def save_sound(sound_array, path):
    soundfile.write(path,
                    sound_array,
                    samplerate=44100,
                    subtype='PCM_16')


def save_design(df, path):
    df.to_csv(path, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputsfolder', metavar='path', required=True,
                        help='the path to the folder for all outputs')
    parser.add_argument('--randomseed', type=int, required=True,
                            help='random generator seed')
    args = parser.parse_args()
    print(args)
    df_design, noise, indicator = create_all_stims(args.randomseed)
    dt = mydatetime(datetime.datetime.now())
    fileroot = os.path.join(args.outputsfolder, dt+'_seed'+str(args.randomseed))

    save_design(df_design, fileroot+'_design.dat')
    save_sound(noise, fileroot+'_noise.wav')
    save_sound(indicator, fileroot+'_indicator.wav')
