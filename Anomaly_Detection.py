import numpy as np
import random
import time
import pandas as pd
import pickle
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import os
import SODA
import data_manipulation as dm
from progress.bar import Bar
import multiprocessing


def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)

def main():
    #-----------------------------------------------------------------------------------#
    #---------------------------------Initiation Part-----------------------------------#

    ####### Variables set by user #######

    # PCA number of components
    N_PCs = 8
    L_N_PCs = 8

    # Range of SODA granularities
    min_granularity = 1

    max_granularity = 2

    # Number of iteration
    iterations = 1

    # Number of process to create in the multiprocessing step
    PROCESSES = 4

    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.

    # Loading data into the code

    ### Background    

    b_name='Reduced_Input_Background_1.csv'

    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]

    ### Signal

    s_name='Reduced_Input_Signal_1.csv'

    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]

    for n_i in range(iterations):


        #------------------------------------------------------------------------------#
        #---------------------------------Main Code------------------------------------#

        # Devide data-set into training and testing sub-sets

        background_train, background_test = train_test_split(background, test_size=0.40, random_state=42)

        # Defining number of events Signal events on online phase.

        signal_online_samples = int(len(background_test)/99)

        # Devide online signal
        reduced_signal, signal_sample_id = dm.divide(signal, 100, signal_online_samples)

        # Nextly, the Signal data processed is saved in the Analised data directory.

        np.savetxt('Analysed_Signal/Reduced_' + s_name,reduced_signal,delimiter=',')
        np.savetxt('Analysed_Signal/Reduced_ID_' + s_name,signal_sample_id,delimiter=',')

        # Concatenating Signal and the Test Background sub-set

        streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

        # Calculates Statistical attributes

        xyz_streaming_data = dm.statistics_attributes(streaming_data,xyz_attributes=True)
        xyz_background_train = dm.statistics_attributes(background_train,xyz_attributes=True)
        xyz_signal = dm.statistics_attributes(signal,xyz_attributes=True)

        # Normalize Features
        norm_xyz_streaming_data = dm.Normalisation(xyz_streaming_data)
        norm_background_train = dm.Normalisation(xyz_background_train)
        norm_divided_signal = dm.Normalisation(xyz_signal)

        # Using Laplace to determine the features to maintain

        maintained_features = dm.laplacian_score(norm_background_train,norm_divided_signal,16)

        # Performing Laplace reduciton 
        laplace_background = dm.laplacian_reduction(norm_background_train,maintained_features)
        laplace_streaming_data = dm.laplacian_reduction(norm_xyz_streaming_data,maintained_features)

        # Calculates PCA and projects the sub-sets 

        proj_xyz_background_train, proj_xyz_streaming_data, xyz_mantained_variation, xyz_attributes_influence = dm.PCA_Projection(norm_background_train,norm_xyz_streaming_data,N_PCs,maintained_features,laplace=False)

        proj_laplace_background, proj_laplace_streaming_data, laplace_mantained_variation, laplace_attributes_influence = dm.PCA_Projection(laplace_background,laplace_streaming_data,L_N_PCs, maintained_features,laplace=True)

        if n_i == 0:
            # Plots PCA results
            dm.PCA_Analysis(xyz_mantained_variation,xyz_attributes_influence,laplace=False)

            dm.PCA_Analysis(laplace_mantained_variation,laplace_attributes_influence,laplace=True)


        print('Creating pool with %d processes\n' % PROCESSES)

        with multiprocessing.Pool(PROCESSES) as pool:

            #
            # Tests

            TASKS = [(dm.SODA_Granularity_Iteration, (proj_xyz_background_train,proj_xyz_streaming_data, gra,len(background_test),n_i,1)) for gra in range(min_granularity, max_granularity + 1)] + [(dm.SODA_Granularity_Iteration, (proj_laplace_background,proj_laplace_streaming_data, gra,len(background_test),n_i,0)) for gra in range(min_granularity, max_granularity + 1)]

            pool.map(calculatestar, TASKS)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       