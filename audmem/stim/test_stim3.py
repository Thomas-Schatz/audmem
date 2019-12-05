# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:07:59 2018

@author: Thomas Schatz & Marianne

Repeated white noise stimuli creation

For a second ERP test.
"""

import numpy as np
import soundfile
import random

# Create 16-bit (signed) 44.1kHz no header mono wavefiles
# (200ms + random uniform jitter in [-50;50] ms) Gaussian white noise followed
# by 200ms reference white noise immediately repeated
# and cycle this for 10 min.
# 1. actual stimuli
# 2. trigger version indicating onsets of repeated noise

# params
fs = 44100  # Hertz
Stim_dur = 0.2  # seconds
ISI_dur = 0.3  # seconds
ITI_min, ITI_max = 0.2, 0.3  # seconds
trial_dur = 2.5 #seconds
n_reps = 5 # reps of noise within trials
block = {"RefRN": 16, "RN": 16, "noise": 32}

# functions
def get_n_samples(dur_s, fs):
    return int(np.floor(fs*dur_s))


def add_silence(list_stim, dur_s, fs):
    n_samp = get_n_samples(fs, dur_s)
    list_stim.append([0.5]*n_samp)


def make_stim_trial(list_stim, ISI_dur):
    list_trial = []
    n_sampISI = get_n_samples(ISI_dur, fs)
    for i in range(n_reps):
        list_trial.append(np.random.randn(n_sampISI))
        list_trial.append(list_stim)
    return np.concatenate(list_trial)

def make_noise_trial(trial_dur):
    list_trial = []
    n_samp = get_n_samples(trial_dur, fs)
    list_trial.append(np.random.randn(n_samp))
    return list_trial

# set seed and generate RefRN
np.random.seed(0)
n_sampStim = get_n_samples(Stim_dur, fs)
REF = np.random.randn(n_sampStim)

# generate one block
trials = []
cond = []
for trial_type in block.keys():
    for n in range(block[trial_type]):
        if trial_type == "RefRN":
            trials.append(make_stim_trial(REF, ISI_dur))
        elif trial_type == "RN":
            RN = np.random.randn(n_sampStim)
            trials.append(make_stim_trial(REF, ISI_dur))
        else:
            trials.append(make_noise_trial(trial_dur))
        cond.append(trial_type)

# adds between trials and end
for t in range(len(trials)):
    ITI_array = np.zeros(shape=(get_n_samples(random.uniform(ITI_min, ITI_max), fs)), dtype=np.float) +.5
    trials.insert(2*t+1, ITI_array)
    print(2*t+1)
# adds silence beginning block
ITI_array = np.zeros(shape=(get_n_samples(random.uniform(ITI_min, ITI_max), fs)), dtype=np.float) +.5
trials.insert(0, ITI_array)



# noise + trigger
np.random.seed(0)
n_sampREF = get_n_samples(REF_dur, fs)
jitter_samps = int(np.floor(fs*jitter_spread))
ref = np.random.randn(n_sampREF)

n_sampISI_base = int(np.floor(fs*ISI_dur))

trigger_dur = .075  # in seconds
trigger_width = get_n_samples(trigger_dur, fs)


n_sampISI_mod = []
n_periods = 0
point = 0
noise, triggers = [], []
while point < total_dur*fs:
  n_periods +=1
  noise = noise + [ref]*(nb_reps+1)
  mod = 2*np.random.randint(jitter_samps) - jitter_samps  # introduce jitter
  n_sampISI_mod.append(mod)
  n_sampISI = n_sampISI_base + mod
  isi = np.random.randn(n_sampISI)
  noise.append(isi)
  total_samp = n_sampREF*(nb_reps+1) + n_sampISI
  trigger = np.zeros(shape=(total_samp), dtype=np.float) +.5
  trigger[:trigger_width] = .8
  triggers.append(trigger)
  point = point + n_sampREF*(nb_reps+1) + n_sampISI

noise = np.concatenate(noise)
triggers = np.concatenate(triggers)

assert len(triggers) == len(noise)

# convert to signed int 16
nb_levels = 65536
mini = -32768
maxi = 32767
# after floor should go from 32767 to -32767
noise = np.floor((nb_levels-1)*noise) - 32767.
noise = noise.astype(np.int16)
triggers = np.floor((nb_levels-1)*triggers) - 32767.
triggers = triggers.astype(np.int16)

# save to wavefile
soundfile.write('/Users/admin/Documents/PhD/data/audmem/noise_v2.wav',
                noise,
                samplerate=44100, subtype='PCM_16')
soundfile.write('/Users/admin/Documents/PhD/data/audmem/trigger_v2.wav',
                triggers,
                samplerate=44100, subtype='PCM_16')
