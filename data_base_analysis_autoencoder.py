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

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def autoencoder_func(data, reduction=11, epochs=200, batch_size=256, activation='exponential', mode='reduction'):
    L, W = data.shape
    
    visible = Input(shape=(W,))
    e = Dense(W*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    e = Dense(W)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    n_bottleneck = round(reduction)
    bottleneck = Dense(n_bottleneck)(e)

    d = Dense(W)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    d = Dense(W*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder_fraud.h5",
                               mode='min', monitor='loss', verbose=0, save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.0001,
            patience=10,
            verbose=0, 
            mode='min',
            restore_best_weights=True)

    output = Dense(W, activation=activation)(d)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[cp, early_stop])
    encoder = Model(inputs=visible, outputs=bottleneck)
    
    if mode == 'reduction':
        reconstructed_data = model.predict(data)
        RMSE = (np.sum((data-reconstructed_data)**2,axis=1)/W)**.5

        encoded_data = encoder.predict(data)
        encoded_data = np.concatenate((encoded_data, RMSE.reshape(-1,1)), axis=1)
        return encoded_data

    elif mode == 'reconstruction':
        reconstructed_data = model.predict(data)

        RMSE = (np.sum((data-reconstructed_data)**2,axis=1)/W)**.5

        encoded_data = np.concatenate((reconstructed_data, RMSE.reshape(-1,1)), axis=1)
        
        return encoded_data
    else:
        print("Mode does not exist, try 'reduction' or 'reconstruction'")


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

def main():
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

    ### Run the ADP to generates groups
    data_analysis = {}
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

        #encoded_streaming_data = autoencoder_func(streaming_data, mode='reconstruction')
        encoded_streaming_data = autoencoder_func(streaming_data, mode='reduction')

        ### Create target

        y =np.ones((len(reduced_background)+len(reduced_signal)))
        y[len(reduced_background):] = -1
        
        ADP_outputs = {}

        print('             .Executing for granularities', gra_list)

        for gra in gra_list:
            print('\n\n             .Iter: {} - Granularity: {}'.format(n_i, gra))
            print('                 .ADP (5th Method)')
            output = ADP_Offline_Granularity_Iteration_5th(encoded_streaming_data, gra)
            ADP_outputs ['granularity_'+str(gra)] = output

        with open('kernel/ADP_outputs_iteration_' + str(n_i) + '.pkl', 'wb') as fp:
            pickle.dump(ADP_outputs, fp)
    
        print('\n        ====Data Processing Complete====\n' )
        print('=*='*17 ) 
        print('      ==== Commencing Data Analysis ====')

        data_clouds_dic = {}
    
        print('         .Formating data_clouds_dic')

        for gra in ADP_outputs:
            idx = ADP_outputs[gra]['IDX']
            u = np.unique(idx)
            aux = {}
            for i in u:
                aux2 = {}
                data = []
                target=[]
                for j in range(len(idx)):
                    if idx[j] == i:
                        data.append(list(encoded_streaming_data[j]))
                        target.append(y[j])

                aux2 ['data'] = data
                aux2 ['target'] = target
                aux['data_cloud_'+str(i)] = aux2
            data_clouds_dic[gra] = aux
                
        with open('kernel/data_clouds_dic_iteration_' + 
                    str(n_i) + '.pkl', 'wb') as fp:
            pickle.dump(data_clouds_dic, fp)

        data_clouds_info = {}

        print('         .Formating data_clouds_info')

        for gra in data_clouds_dic:
            aux = {
                'n_data_clouds':len(data_clouds_dic[gra])
            }
            aux2={}
            aux3={}
            for dc in data_clouds_dic[gra]:
                target = data_clouds_dic[gra][dc]['target']
                aux2[dc] = len(target)
                aux3[dc] = target.count(-1)
            aux['n_events_p_dc'] = aux2
            aux['n_anom_p_dc'] = aux3
            data_clouds_info[gra] = aux
        
        with open('kernel/data_clouds_info_iteration_' + 
                    str(n_i) + '.pkl', 'wb') as fp:
            pickle.dump(data_clouds_info, fp)

        print('\n        ====Data Analysis Complete====\n' )
        print('=*='*17 )

if __name__ == '__main__':
    main() 