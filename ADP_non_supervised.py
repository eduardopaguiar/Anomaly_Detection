import time
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import data_manipulation as dm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from tsfresh.utilities.dataframe_functions import impute
import ADP

def ADP_Offline_Granularity_Iteration_5th(streaming, gra):
    begin = dm.datetime.now()

    ##################################
    ##### ----- STATIC ADP ----- #####
    ##### ---------------------- #####

    Input = {'data': streaming,
             'granularity': gra,
             'distancetype': 'euclidean'}
            
        
    ADP_streaming_output = ADP.ADP(Input, 'Offline')

    # Computing the number of clouds
    ADP_streaming_output['n_data_clouds'] = max(ADP_streaming_output['IDX']) + 1
    
    return ADP_streaming_output

if __name__ == '__main__':
    ##########################################################
    # ------------------------------------------------------ #
    # --------------------- INITIATION --------------------- #
    # ------------------------------------------------------ #
    ##########################################################
    ### Define User Variables ###

    # List of Granularities
    gra_list = [i for i in range(1,11)]

    # Number of Iterations
    iterations = 33

    # Number of events
    total = 10000

    # Number of Data-set divisions
    windows = 100

    # Percentage of background samples on the testing phase
    background_percent = 0.99

    # Percentage of samples on the training phase
    test_size = 0.3
    
    ##########################################################
    # ------------------------------------------------------ #
    # ----------------------- LOADING ---------------------- #
    # ------------------------------------------------------ #
    ##########################################################
    # Firstly the model loads the background and signal data, 
    # then it removes the attributes first string line, which 
    # are the column names, in order to avoid NaN values in
    # the array.

    print('         ==== Commencing Initiation ====\n')

    ### Background    
    b_name='Input_Background_1.csv'
    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]
    print("     .Background Loaded..." )
    print("     .Background shape: {}".format(background.shape))

    ### Signal
    s_name='Input_Signal_1.csv'
    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]
    print("     .Signal Loaded...")
    print("     .Signal shape: {}\n".format(signal.shape))

    print('\n          ==== Initiation Complete ====\n')
    print('=*='*17 )
    print('      ==== Commencing Data Processing ====')

    ADP_outputs = {}
    for n_i in range(iterations):
        print('\n     => Iteration Number', (n_i+1) )

        # Divide data-set into training and testing sub-sets
        print('         .Dividing training and testing sub-sets')

        b_total = int(total*background_percent)
        reduced_background, _ = dm.divide(background, windows, b_total)

        # Defining number of events Signal events on online phase.
        signal_online_samples = int(total - b_total)
        reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

        print('         .Selecting Signal on the following porpotion:')
        print('             .{}% Background samples'.format(int(background_percent*100)))
        print('             .{}% Signal samples'.format(int((1-background_percent)*100)))

        # Concatenating Signal and the Test Background sub-set
        streaming_data_raw = np.concatenate((reduced_background,reduced_signal), axis=0)
        print("             .Online shape: {}\n".format(streaming_data_raw.shape))

        # Normalize Data
        print('         .Normalizing Data')
        streaming_data = normalize(streaming_data_raw,norm='max',axis=0)
        
        aux = {}
        
        print('             .Executing for granularities', gra_list)
        for gra in gra_list:
            print('\n\n             .Iter: {} - Granularity: {}'.format(n_i, gra))
            print('                 .ADP (5th Method)')
            output = ADP_Offline_Granularity_Iteration_5th(streaming_data, gra)
            aux ['granularity_'+str(gra)] = output
        ADP_outputs['iteration_'+str(n_i)] = aux                  
        
        ### Doc 
        ### ADP_outputs: 
        ### >>> ADP_outputs['iteration_0']['granularity_1'].keys()
        ### >>> dict_keys(['centre', 'IDX', 'Param', 'n_data_clouds'])
        
