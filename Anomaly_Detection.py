
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

#-------------------------------------------------------------------------------------#
#---------------------------------Initiation Part-------------------------------------#

####### Variables set by user #######

# PCA number of components
N_PCs = 8

# Range of SODA granularities
min_granularity = 20
max_granularity = 50

# Number of iteration
n_i = 33

# Firstly the model loads the background and signal data, then it removes the 
# attributes first string line, in order to avoid NaN values in the array.

# Loading data into the code

### Background    

b_name='Input_Background_1.csv'

background = np.genfromtxt(b_name, delimiter=',')
background = background[1:,:]

### Signal

s_name='Input_Signal_1.csv'

signal = np.genfromtxt(s_name, delimiter=',')
signal = signal[1:,:]

# Devide data-set into training and testing sub-sets

background_train, background_test = train_test_split(background, test_size=0.40, random_state=42)

# Defining number of events Signal events.

signal_samples = int(len(background_test)/99)

#-------------------------------------------------------------------------------------#
#-------------------------------------Main Code---------------------------------------#

# Iniciates progress bar
bar = Bar(('Progess:'), max=n_i)
bar.start()

for i in range(n_i):
    # all attributes with nomalisation after PCA
    # Devide online signal
    reduced_signal, signal_sample_id = dm.divide(signal, 100, signal_samples)

    # Creates a label for the analyses part

    # Nextly, the Signal data processed is saved in the Analised data directory.

    np.savetxt('Analysed_Signal/Reduced_' + s_name,reduced_signal,delimiter=',')
    np.savetxt('Analysed_Signal/Reduced_ID_' + s_name,signal_sample_id,delimiter=',')

    # Concatenating Signal and the Test Background sub-set

    streaming_data = np.concatenate((background_test,reduced_signal), axis=0)

    # Calculates Statistical attributes

    background_train_stat = dm.statistics_attributes(background_train,xyz_attributes=False)
    streaming_data = dm.statistics_attributes(streaming_data,xyz_attributes=False)

    # Calculates PCA and projects the sub-sets 

    proj_background_train, proj_streaming_data, mantained_variation, attributes_influence = dm.PCA_Projection(background_train_stat,streaming_data,N_PCs,norm=True)

    if i == 0:
        # Plots PCA results

        dm.PCA_Analysis(mantained_variation,attributes_influence,norm=True,xyz_attributes=False)

        # Performes SODA interactively whith the given granularities

    dm.SODA_Granularity_Iteration(proj_background_train,proj_streaming_data,max_granularity,min_granularity,len(background_test),i,norm=True,xyz_attributes=False)

    bar.next()
bar.finish()