# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:07:59 2018

@author: Thomas Schatz

Repeated white noise stimuli creation

For a first ERP test.
"""

import numpy as np
import soundfile

# Create 16-bit (signed) 44.1kHz no header mono wavefiles
# (200ms + random uniform jitter in [-50;50] ms) Gaussian white noise followed
# by 200ms reference white noise immediately repeated
# and cycle this for 10 min.
# 1. actual stimuli
# 2. trigger version indicating onsets of repeated noise

# params
fs = 44100  # Hertz
REF_dur = 0.2  # seconds
nb_reps = 1  # there will be nb_reps+1 presentations total
ISI_dur = 0.2  # seconds
total_dur = 1200  # seconds
jitter_spread = 0.15  # seconds

# noise + trigger
np.random.seed(0)
n_sampREF = int(np.floor(fs*REF_dur))
jitter_samps = int(np.floor(fs*jitter_spread))
ref = np.random.randn(n_sampREF)

n_sampISI_base = int(np.floor(fs*ISI_dur))

trigger_dur = .05  # in seconds
trigger_width = int(np.floor(trigger_dur*fs))


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
