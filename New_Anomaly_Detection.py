import numpy as np
import data_manipulation as dm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

import ADP

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
    # Firstly the model loads the background and signal data, then it removes the 
    # attributes first string line, in order to avoid NaN values in the array.
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

    for n_i in range(iterations):
        print('\n     => Iteration Number', (n_i+1) )

        # Divide data-set into training and testing sub-sets
        print('         .Dividing training and testing sub-sets')
        divided_background, _ = dm.divide(background, windows, total)

        test = int(total*test_size)
        b_test = int(test*background_percent)
        static_data_raw, background_test = train_test_split(divided_background, test_size=test_size, random_state=42)
        background_test, _ = dm.divide(background_test, windows, b_test)

        # Defining number of events Signal events on online phase.
        signal_online_samples = int(test - b_test)
        reduced_signal, _ = dm.divide(signal, windows, signal_online_samples)

        print('         .Selecting Signal on the following porpotion:')
        print('             .{}% Background samples'.format(int(background_percent*100)))
        print('             .{}% Signal samples'.format(int((1-background_percent)*100)))
        print('             .{:9d} of Background samples (Offline)'.format(int(total*(1-test_size))))
        print('             .{:9d} of Background samples (Online)'.format(int(b_test)) )
        print('             .{:9d} of Signal samples (Online)'.format(int(signal_online_samples)))

        # Concatenating Signal and the Test Background sub-set
        streaming_data_raw = np.concatenate((background_test,reduced_signal), axis=0)
        print("             .Offline shape: {}".format(static_data_raw.shape))
        print("             .Online shape: {}\n".format(streaming_data_raw.shape))

        # Normalize Data
        print('         .Normalizing Data')
        norm = Normalizer(norm='max').fit(static_data_raw.T)
        static_data = norm.transform(static_data_raw.T).T
        streaming_data = norm.transform(streaming_data_raw.T).T

        print('             .Executing for granularities', gra_list)
        for gra in gra_list:
            print('\n\n             .Iter: {} - Granularity: {}'.format(n_i, gra))
            print('                 .ADP (First Method)')
            dm.NEW_ADPOffline_Granularity_Iteration_first(static_data, streaming_data, gra, b_test, n_i)
            print('\n                 .ADP (Second Method)')
            dm.NEW_ADPOffline_Granularity_Iteration_second(static_data, streaming_data, gra, b_test, n_i)
            print('\n                 .ADP (Third Method)')
            dm.NEW_ADPOffline_Granularity_Iteration_third(static_data, streaming_data, gra, b_test, n_i)

    print('\n        ====Data Processing Complete====\n' )
    print('=*='*17 ) 