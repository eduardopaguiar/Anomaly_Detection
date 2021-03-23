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
    N_PCs = 5

    # List of granularities 
    gra_list = [6,8,10,12] 

    # Number of iteration
    iterations = 16

    # Number of process to create in the multiprocessing step
    PROCESSES = 4

    # Number of Data-set divisions
    total = 25000

    # Number of Data-set divisions
    windows = 100

    # Percentage of background samples on the testing phase
    background_percent = 99

    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.

    # Using multiprocess to load the data-sets into the code

    print('         ==== Commencing Initiation ====\n', file=open("log_file.txt", "a+"))

    ### Background    
    #b_name='/AtlasDisk/user/pestana/Input/Input_Background_1.csv'
    b_name='Input_Background_1.csv'

    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]
    background, _ = dm.divide(background, windows, total)
    print("     .Background Loaded...", file=open("log_file.txt", "a"))

    ### Signal
    #s_name='/AtlasDisk/user/pestana/Input/Input_Signal_1.csv'
    s_name='Input_Signal_1.csv'

    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]
    print("     .Signal Loaded...", file=open("log_file.txt", "a"))

    print('\n          ==== Initiation Complete ====\n', file=open("log_file.txt", "a"))
    print('=*='*17, file=open("log_file.txt", "a"))

    print('      ==== Commencing Data Processing ====', file=open("log_file.txt", "a"))
    
    """
    with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(model, (n_i,background,background_percent,signal,windows,N_PCs,gra_list, total)) for n_i in range(iterations)]

        print('             .Executing SODA for granularities', gra_list, file=open("log_file.txt", "a"))

        pool.map(calculatestar, TASKS)
    """
    for n_i in range (iterations):
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

        print('         .Selecting Signal on the following porpotion:', file=open("log_file.txt", "a"))
        print('             .' + str(background_percent) + '% Background samples', file=open("log_file.txt", "a"))
        print('             .' + str(100-background_percent) + '% Signal samples', file=open("log_file.txt", "a"))
        print('             .{:9d} of Background samples (Offline)'.format(int(total*(1-test_size))), file=open("log_file.txt", "a"))
        print('             .{:9d} of Background samples (Online)'.format(int(b_test)), file=open("log_file.txt", "a"))
        print('             .{:9d} of Signal samples (Online)'.format(int(signal_online_samples)), file=open("log_file.txt", "a"))

        reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

        # Nextly, the Signal data processed is saved in the Analised data directory.

        #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_iteration_' + str(n_i) + '_' + s_name,reduced_signal,delimiter=',')
        #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_ID_iteration_' + str(n_i) + '_' + s_name,signal_sample_id,delimiter=',')

        # Concatenating Signal and the Test Background sub-set

        streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

        # Normalize Data

        norm_background_train = preprocessing.normalize(background_train)
        norm_streaming_data = preprocessing.normalize(streaming_data)

        #print('         .Normalizing Data', file=open("log_file.txt", "a"))

        # Calculates Statistical attributes

        print('         .Calculating statistical attributes', file=open("log_file.txt", "a"))

        xyz_streaming_data = dm.statistics_attributes(norm_streaming_data)
        xyz_background_train = dm.statistics_attributes(norm_background_train)
        #xyz_streaming_data = dm.statistics_attributes(streaming_data)
        #xyz_background_train = dm.statistics_attributes(background_train)

        #xyz_signal = dm.statistics_attributes(signal)
        #xyz_background = dm.statistics_attributes(background)
        
        #transformer = preprocessing.Normalizer().fit(np.vstack((xyz_signal,xyz_background)))
        
        #xyz_background = transformer.transform(xyz_background)
        #xyz_signal = transformer.transform(xyz_signal)

        #np.savetxt('xyz_reduced_signal_norm.csv',xyz_signal,delimiter=',')
        #np.savetxt('xyz_background_norm.csv',xyz_background,delimiter=',')

        # Normalize Features

        #print('         .Normalizing Features', file=open("log_file.txt", "a"))

        # Calculates PCA and projects the sub-sets 

        print('         .Calculating PCA:', file=open("log_file.txt", "a"))

        proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(xyz_background_train,xyz_streaming_data,N_PCs)
        #proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

        # Plots PCA results

        print('         .Ploting PCA results', file=open("log_file.txt", "a"))

        dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence)

        #proj_xyz_background_train = preprocessing.normalize(proj_xyz_background_train)
        #proj_xyz_streaming_data = preprocessing.normalize(proj_xyz_streaming_data)

        print('         .Running SODA on base granularity', file=open("log_file.txt", "a"))
        dm.SODA_Granularity_Iteration(proj_xyz_background_train,proj_xyz_streaming_data, 1,len(background_test),n_i)

        print('         .Creating pool with %d processes:' % PROCESSES, file=open("log_file.txt", "a"))

        with multiprocessing.Pool(PROCESSES) as pool:

            TASKS = [(dm.SODA_Granularity_Iteration, (proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i)) for gra in gra_list]

            print('             .Executing SODA for granularities', gra_list, file=open("log_file.txt", "a"))

            pool.map(calculatestar, TASKS)
 
        
    print('\n        ====Data Processing Complete====\n', file=open("log_file.txt", "a"))
    print('=*='*17, file=open("log_file.txt", "a"))

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

    print('         .Selecting Signal on the following porpotion:', file=open("log_file.txt", "a"))
    print('             .' + str(background_percent) + '% Background samples', file=open("log_file.txt", "a"))
    print('             .' + str(100-background_percent) + '% Signal samples', file=open("log_file.txt", "a"))
    print('             .{:9d} of Background samples (Offline)'.format(int(total*(1-test_size))), file=open("log_file.txt", "a"))
    print('             .{:9d} of Background samples (Online)'.format(int(b_test)), file=open("log_file.txt", "a"))
    print('             .{:9d} of Signal samples (Online)'.format(int(signal_online_samples)), file=open("log_file.txt", "a"))

    reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

    # Nextly, the Signal data processed is saved in the Analised data directory.

    #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_iteration_' + str(n_i) + '_' + s_name,reduced_signal,delimiter=',')
    #np.savetxt('/AtlasDisk/user/pestana/Output/Analysed_Signal/Reduced_ID_iteration_' + str(n_i) + '_' + s_name,signal_sample_id,delimiter=',')

    # Concatenating Signal and the Test Background sub-set

    streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

    # Normalize Data

    norm_background_train = preprocessing.normalize(background_train)
    norm_streaming_data = preprocessing.normalize(streaming_data)

    #print('         .Normalizing Data', file=open("log_file.txt", "a"))

    # Calculates Statistical attributes

    print('         .Calculating statistical attributes', file=open("log_file.txt", "a"))

    xyz_streaming_data = dm.statistics_attributes(norm_streaming_data)
    xyz_background_train = dm.statistics_attributes(norm_background_train)
    #xyz_streaming_data = dm.statistics_attributes(streaming_data)
    #xyz_background_train = dm.statistics_attributes(background_train)

    #xyz_signal = dm.statistics_attributes(signal)
    #xyz_background = dm.statistics_attributes(background)
    
    #transformer = preprocessing.Normalizer().fit(np.vstack((xyz_signal,xyz_background)))
    
    #xyz_background = transformer.transform(xyz_background)
    #xyz_signal = transformer.transform(xyz_signal)

    #np.savetxt('xyz_reduced_signal_norm.csv',xyz_signal,delimiter=',')
    #np.savetxt('xyz_background_norm.csv',xyz_background,delimiter=',')

    # Normalize Features

    #print('         .Normalizing Features', file=open("log_file.txt", "a"))

    # Calculates PCA and projects the sub-sets 

    print('         .Calculating PCA:', file=open("log_file.txt", "a"))

    proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(xyz_background_train,xyz_streaming_data,N_PCs)
    #proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_xyz_background_train,norm_xyz_streaming_data,N_PCs)

    # Plots PCA results

    print('         .Ploting PCA results', file=open("log_file.txt", "a"))

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
    
        
    print('\n        ====Data Processing Complete====\n', file=open("log_file.txt", "a"))
    print('=*='*17, file=open("log_file.txt", "a"))
            

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       
