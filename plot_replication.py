# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:57:30 2021

@author: drewj
"""
""" Outline:
0. Initialize an empty Numpy ndarray to hold pev for each neuron in each area.  
Different datafiles have inconsistent numbers of neurons that were obtained from each brain area, 
so you can't really allocate the array for all neurons at once, you have to do some kinda ugly concatenation below.

1. Loop through all matlab files
    a. load data and get spike times and trial info and unit info
    b. use rate() to find spike rate and rate bins
    c. compute spike density and timepoints?
    d. use neural_info to calculate percent explained variance
    e. FIRST STEP: avg over each neuron in file
    e. loop through each brain area
        1. use 'area' field in unit info to find all neurons in file from that area
        2. concatenate pev for these neurons into the array holding pev's for all neurons from given brain area (see np.concatenate)

2. Loop through each brain area
    1. average over all neurons in given area
    2. plot pev vs. time for given area
"""
import neural_analysis.spikes as spk
from neural_analysis.info import neural_info
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_analysis.matIO import loadmat
#########################################
####      Helper Functions       ########
#########################################
def avg_pev(num_units, rate_bins, pev):
    """
    Parameters
    ----------
    num_units : Int
        Number of units in file session
    rate_bins : Int
        number of timepoints taken
    pev : Tuple
        pev after using neural_info. Should be in shape (1,units,time)

    Returns
    -------
    ndarray of PEV values average across all units in a datafile

    """
    #Empty numpy ndarry to hold pev for each neuron in each area
    pev_avg = np.zeros(rate_bin_centers.size)
    
    #avg pev across all neurons in file
    #Steps: new nparr with pev at 1 timept for all units. sum array, divide by length, add to pev_avg array
    num_units=pev.shape[1]
    allunits=np.zeros(num_units)
    #print(rate_bin_centers.size)
    for time in range(rate_bin_centers.size):
        for unit in range (num_units):
           # print("unit",unit)
            allunits[unit]=np.squeeze(pev[:,unit,:])[time]
        avg=np.sum(allunits)/allunits.size
        pev_avg[time]=avg
    

#########################################
#####   Load Data From File      ########
#########################################
#load data, get spike times, trial info, unit info
filename = r'/Users/drewj/Documents/millerUROP/millerdata/ZC20130911.mat'
trial_info,spike_times, unit_info = loadmat(filename, variables=['trialInfo','spikeTimes','unitInfo'])


#########################################
####   Extract unit and neuron data  ####
#########################################

#use rate() to find spike rate and rate bins
spike_rate, rate_bins = spk.rate(spike_times, method='bin', lims=[-0.5,2.5], width=50e-3, step=10e-3)
rate_bin_centers = np.mean(rate_bins,axis=1)    #represents timepoints

print("Binned spike rate (trials,units,time bins): ", spike_rate.shape)

# Compute spike density and timepoints -> with 50 ms Gaussian kernel and downsample to 100 Hz (10 ms)?
spike_density, timepts = spk.rate(spike_times, method='density', lims=[-0.5,2.5], kernel='gaussian', width=50e-3,
                              smp_rate=1000, downsmp=10)

#use neural_info to calculate pev
labels = trial_info['sample']
pev = neural_info(labels, spike_rate, axis=0, method='pev', model='anova1')
print("PEV (1,units,time): ", pev.shape)


#########################################
####    Avg neurons in file      ########
#########################################

#Empty numpy ndarry to hold pev for each neuron in each area
pev_avg = np.zeros(rate_bin_centers.size)

#avg pev across all neurons in file
#Steps: new nparr with pev at 1 timept for all units. sum array, divide by length, add to pev_avg array
num_units=pev.shape[1]
allunits=np.zeros(num_units)
#print(rate_bin_centers.size)
for time in range(rate_bin_centers.size):
    for unit in range (num_units):
       # print("unit",unit)
        allunits[unit]=np.squeeze(pev[:,unit,:])[time]
    avg=np.sum(allunits)/allunits.size
    pev_avg[time]=avg
    



#########################################
########    Plotting            #########
#########################################

#plot pev vs time
plt.figure()
for unit in range(pev.shape[1]):
    plt.subplot(2,2,unit+1)
    plt.plot(rate_bin_centers, np.squeeze(pev[:,unit,:]))   # Squeeze out singleton axis
    plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
    if unit == 0: plt.ylabel('PEV')
    plt.title("Unit %d" % unit)
    plt.xlabel('Time (s)')
plt.subplot(2,2,3)
plt.plot(rate_bin_centers,pev_avg)
plt.title("Avg")
plt.xlabel('Time (s)')
plt.tight_layout()
    
plt.savefig('spikerates.png')



        





