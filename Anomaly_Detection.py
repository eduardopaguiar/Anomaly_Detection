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
    gra_list = [14] 

    # Number of iteration
    iterations = 4

    # Number of process to create in the multiprocessing step
    PROCESSES = 1

    # Number of Data-set divisions
    windows = 100

    # Percentage of background samples on the testing phase
    background_percent = 99

    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.

    # Using multiprocess to load the data-sets into the code

    print('         ==== Commencing Initiation ====\n', file=open("log_file.txt", "a"))

    ### Background    
    b_name='/AtlasDisk/user/pestana/Input/Input_Background_1.csv'

    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]
    background, _ = dm.divide(background, 100, 100000)
    print("     .Background Loaded...", file=open("log_file.txt", "a"))

    ### Signal
    s_name='/AtlasDisk/user/pestana/Input/Input_Signal_1.csv'

    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]
    print("     .Signal Loaded...", file=open("log_file.txt", "a"))

    print('\n          ==== Initiation Complete ====\n', file=open("log_file.txt", "a"))
    print('=*='*17, file=open("log_file.txt", "a"))

    print('      ==== Commencing Data Processing ====', file=open("log_file.txt", "a"))

    for n_i in range(iterations):

        print('\n     => Iteration Number', (n_i+1), file=open("log_file.txt", "a"))

        # Devide data-set into training and testing sub-sets

        print('         .Deviding training and testing sub-sets', file=open("log_file.txt", "a"))

        background_train, background_test = train_test_split(background, test_size=0.30, random_state=42)

        # Defining number of events Signal events on online phase.

        signal_online_samples = int(len(background_test)/background_percent)

        # Devide online signal

        print('         .Selecting Signal on the following porpotion:', file=open("log_file.txt", "a"))
        print('             .' + str(background_percent) + '% Background samples', file=open("log_file.txt", "a"))
        print('             .' + str(100-background_percent) + '% Signal samples', file=open("log_file.txt", "a"))

        reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

        # Nextly, the Signal data processed is saved in the Analised data directory.
    
        #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_iteration_' + str(n_i) + '_' + s_name,reduced_signal,delimiter=',')
        #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_ID_iteration_' + str(n_i) + '_' + s_name,signal_sample_id,delimiter=',')
    
        # Concatenating Signal and the Test Background sub-set

        streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

        # Normalize Data

        print('         .Normalizing Data', file=open("log_file.txt", "a"))

        # Calculates Statistical attributes

        print('         .Calculating statistical attributes', file=open("log_file.txt", "a"))

        xyz_streaming_data = dm.statistics_attributes(streaming_data)
        xyz_background_train = dm.statistics_attributes(background_train)

        # Normalize Features

        print('         .Normalizing Features', file=open("log_file.txt", "a"))

        norm_xyz_background_train,norm_xyz_streaming_data = dm.Normalisation(xyz_background_train,xyz_streaming_data)

        # Calculates PCA and projects the sub-sets 

        print('         .Calculating PCA:', file=open("log_file.txt", "a"))

        proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

        # Plots PCA results

        print('         .Ploting PCA results', file=open("log_file.txt", "a"))

        dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence)


        print('         .Running SODA on base granularity', file=open("log_file.txt", "a"))
        dm.SODA_Granularity_Iteration(proj_xyz_background_train,proj_xyz_streaming_data, 1,len(background_test),n_i)

        print('         .Creating pool with %d processes:' % PROCESSES, file=open("log_file.txt", "a"))
    
        with multiprocessing.Pool(PROCESSES) as pool:

            TASKS = [(dm.SODA_Granularity_Iteration, (proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)) for gra in gra_list]
    
            print('             .Executing SODA for granularities', gra_list, file=open("log_file.txt", "a"))

            pool.map(calculatestar, TASKS)

        
    print('\n        ====Data Processing Complete====\n', file=open("log_file.txt", "a"))
    print('=*='*17, file=open("log_file.txt", "a"))
            

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       
