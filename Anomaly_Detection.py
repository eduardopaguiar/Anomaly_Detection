import numpy as np
import pandas as pd
import pickle
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import os
import SODA
import data_manipulation as dm
import multiprocessing
from sklearn.utils.validation import check_array
import sys

#-------------------------Main Code--------------------------#

def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)


def main():
    #------------------------------------------------------------#
    #-------------------Initiation Part--------------------------#

    ####### Variables set by user #######

    # PCA number of components
    N_PCs = 8

    # List of granularities 
    gra_list = [i for i in range(1,31)] 

    # Number of iteration
    iterations = 3

    # Number of process to create in the multiprocessing step
    PROCESSES = 4

    # Number of Data-set divisions
    windows = 100

    # Percentage of background samples on the testing phase
    background_percent = 99

    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.

    # Using multiprocess to load the data-sets into the code

    print('         ==== Commencing Initiation ====\n')

    ### Background    
    b_name='Input_Background_1.csv'

    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]

    print("     .Background Loaded...")

    ### Signal
    s_name='Input_Signal_1.csv'

    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]

    print("     .Signal Loaded...")

    print('\n          ==== Initiation Complete ====\n')
    print('=*='*17)

    print('      ==== Commencing Data Processing ====')

    for n_i in range(iterations):

        print('\n     => Iteration Number', (n_i+1))

        # Devide data-set into training and testing sub-sets

        print('         .Deviding training and testing sub-sets')

        background_train, background_test = train_test_split(background, test_size=0.40, random_state=42)

        # Defining number of events Signal events on online phase.

        signal_online_samples = int(len(background_test)/background_percent)

        # Devide online signal

        print('         .Selecting Signal on the following porpotion:')
        print('             .' + str(background_percent) + '% Background samples')
        print('             .' + str(100-background_percent) + '% Signal samples')

        reduced_signal, signal_sample_id = dm.divide(signal, windows, signal_online_samples)

        # Nextly, the Signal data processed is saved in the Analised data directory.
    
        np.savetxt('Analysed_Signal/Reduced_iteration_' + str(n_i) + '_' + s_name,reduced_signal,delimiter=',')
        np.savetxt('Analysed_Signal/Reduced_ID_iteration_' + str(n_i) + '_' + s_name,signal_sample_id,delimiter=',')
    
        # Concatenating Signal and the Test Background sub-set

        streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

        # Normalize Data

        print('         .Normalizing Data')
        
        norm_streaming_data = dm.Normalisation(streaming_data)
        norm_background_train = dm.Normalisation(background_train)

        # Calculates Statistical attributes

        print('         .Calculating statistical attributes')

        xyz_streaming_data = dm.statistics_attributes(norm_streaming_data)
        xyz_background_train = dm.statistics_attributes(norm_background_train)

        # Normalize Features

        print('         .Normalizing Features')

        norm_xyz_streaming_data = dm.Normalisation(xyz_streaming_data)
        norm_xyz_background_train = dm.Normalisation(xyz_background_train)

        # Calculates PCA and projects the sub-sets 

        print('         .Calculating PCA:')

        proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

        # Plots PCA results

        print('         .Ploting PCA results')

        dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence)

        
        for gra in gra_list:
            dm.SODA_Granularity_Iteration(proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)
        '''

        print('         .Creating pool with %d processes:' % PROCESSES)
    
        with multiprocessing.Pool(PROCESSES) as pool:

            TASKS = [(dm.SODA_Granularity_Iteration, (proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)) for gra in gra_list]
    
            print('             .Executing SODA for granularities', gra_list)

            pool.map(calculatestar, TASKS)'''
            
    print('\n        ====Data Processing Complete====\n')
    print('=*='*17)
            

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       
