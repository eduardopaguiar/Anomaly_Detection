import numpy as np
import pandas as pd
import pickle
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, Normalizer
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
    N_PCs = 6

    # Number of Data-set divisions
    total = 10000

    # Number of Data-set divisions
    windows = 100

    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.

    # Using multiprocess to load the data-sets into the code

    print('         ==== Commencing Initiation ====\n', file=open("log_file.txt", "a+"))

    ### Background    
    #b_name='/AtlasDisk/user/pestana/Input/Input_Background_1.csv'
    b_name='Input_Background_1.csv'

    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]
    print("     .Background Loaded...", file=open("log_file.txt", "a"))

    ### Signal
    #s_name='/AtlasDisk/user/pestana/Input/Input_Signal_1.csv'
    s_name='Input_Signal_1.csv'

    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]
    print("     .Signal Loaded...", file=open("log_file.txt", "a"))

    # Devide inputs
    
    #background, _ = dm.divide(background, windows, int(total/2))

    #signal, _ = dm.divide(signal, windows, int(total/2))

    # Normalize Data
    
    #scaler = Normalizer(norm='max').fit(background_train.T)
    #norm_background_train = scaler.transform(background_train.T).T
    #norm_streaming_data = scaler.transform(streaming_data.T).T

    # Calculates Statistical attributes

    print('         .Calculating statistical attributes', file=open("log_file.txt", "a"))

    #xyz_signal = dm.statistics_attributes(signal)
    #xyz_background = dm.statistics_attributes(background)
    xyz_signal = dm.statistics_attributes(signal)
    xyz_background = dm.statistics_attributes(background)

    # Calculates PCA and projects the sub-sets 

    proj_xyz_background, proj_xyz_signal, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(xyz_background,xyz_signal,N_PCs)
    #proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

    # Plots PCA results

    dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence)

    #proj_xyz_background_train = preprocessing.normalize(proj_xyz_background_train)
    #proj_xyz_streaming_data = preprocessing.normalize(proj_xyz_streaming_data)

    SODA.SODA_plot (proj_xyz_background,proj_xyz_signal)
    """
    print('         .Creating pool with %d processes:' % PROCESSES, file=open("log_file.txt", "a"))

    with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(SODA.SODA_plot, (proj_xyz_background,proj_xyz_signal, gra)) for gra in gra_list]

        pool.map(calculatestar, TASKS)
    
    
    with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(model, (n_i,background,background_percent,signal,windows,N_PCs,gra_list, total)) for n_i in range(iterations)]

        print('             .Executing SODA for granularities', gra_list, file=open("log_file.txt", "a"))

        pool.map(calculatestar, TASKS)
    """
 
        

def model(n_i,background,background_percent,signal,windows,N_PCs,gra_list, total):

    print('\n     => Iteration Number', (n_i+1), file=open("log_file.txt", "a"))

    # Devide data-set into training and testing sub-sets

    print('         .Dividing training and testing sub-sets', file=open("log_file.txt", "a"))

    test_size = 0.3
    test = int(total*test_size)
    b_test = int(test*background_percent/100)
    background_train, background_test = train_test_split(background, test_size=0.30, random_state=42)
    background_test, _ = dm.divide(background_test, windows, b_test)

    # Defining number of events Signal events on online phase.

    signal_online_samples = int(test - b_test)

    # Devide online signal

    reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

    # Concatenating Signal and the Test Background sub-set

    streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

    # Normalize Data

    norm_background_train = preprocessing.normalize(background_train)
    norm_streaming_data = preprocessing.normalize(streaming_data)

    # Calculates Statistical attributes

    xyz_streaming_data = dm.statistics_attributes(norm_streaming_data)
    xyz_background_train = dm.statistics_attributes(norm_background_train)
    #xyz_streaming_data = dm.statistics_attributes(streaming_data)
    #xyz_background_train = dm.statistics_attributes(background_train)

    # Calculates PCA and projects the sub-sets 

    proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(xyz_background_train,xyz_streaming_data,N_PCs)
    #proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

    # Plots PCA results

    dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence)

    #proj_xyz_background_train = preprocessing.normalize(proj_xyz_background_train)
    #proj_xyz_streaming_data = preprocessing.normalize(proj_xyz_streaming_data)
    
    for gra in gra_list:
        dm.SODA_Granularity_Iteration(proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)
    
    """
    print('         .Running SODA on base granularity', file=open("log_file.txt", "a"))
    dm.SODA_Granularity_Iteration(proj_xyz_background_train,proj_xyz_streaming_data, 1,len(background_test),n_i)

    print('         .Creating pool with %d processes:' % PROCESSES, file=open("log_file.txt", "a"))

    with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(dm.SODA_Granularity_Iteration, (proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)) for gra in gra_list]

        print('             .Executing SODA for granularities', gra_list, file=open("log_file.txt", "a"))

        pool.map(calculatestar, TASKS)
    """
            

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       
