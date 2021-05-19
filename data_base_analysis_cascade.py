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
    total = 143

    # Number of Data-set divisions
    windows = 334

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

    # Percentage of background samples to divide the data-set
    dat_set_percent = total/len(background)

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

        _,reduced_background = train_test_split(background, test_size=dat_set_percent)

        test = total*test_size
        b_test = test*background_percent
        b_test_percent = b_test / test

        background_seed, streaming_background = train_test_split(reduced_background, test_size=test_size)

        _,streaming_background = train_test_split(reduced_background, test_size=b_test_percent)

        # Defining number of events Signal events on online phase.
        _,reduced_signal = train_test_split(signal, test_size=dat_set_percent*(1 - b_test_percent))

        # Concatenating Signal and the Test Background sub-set
        streaming_data = np.vstack((background_seed, streaming_background,reduced_signal))

        print("             .Seed shape: {}".format(background_seed.shape))
        print("             .Streaming shape: {}\n".format(streaming_data.shape))
        print("             .Streaming Background shape: {}\n".format(streaming_background.shape))
        print("             .Streaming Signal shape: {}\n".format(reduced_signal.shape))

        # Normalize Data
        print('         .Normalizing Data')
        streaming_data = normalize(streaming_data,norm='max',axis=0)

        ### Create target

        y =np.ones((len(streaming_data)))
        y[len(streaming_background):] = -1
        y[len(background_seed):] = 0
        
        ADP_outputs = {}

        print('             .Executing for granularities', gra_list)

        for gra in gra_list:
            print('\n\n             .Iter: {} - Granularity: {}'.format(n_i, gra))
            print('                 .ADP (5th Method)')
            output = ADP_Offline_Granularity_Iteration_5th(streaming_data, gra)
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
                        data.append(list(streaming_data[j]))
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
            aux4={}
            for dc in data_clouds_dic[gra]:
                target = data_clouds_dic[gra][dc]['target']
                aux2[dc] = len(target)
                aux3[dc] = target.count(-1)
                aux4[dc] = target.count(0)
            aux['n_events_p_dc'] = aux2
            aux['n_anom_p_dc'] = aux3
            aux['n_seed_p_dc'] = aux4
            data_clouds_info[gra] = aux
        
        with open('kernel/data_clouds_info_iteration_' + 
                    str(n_i) + '.pkl', 'wb') as fp:
            pickle.dump(data_clouds_info, fp)

        print('\n        ====Data Analysis Complete====\n' )
        print('=*='*17 )

if __name__ == '__main__':
    main() 