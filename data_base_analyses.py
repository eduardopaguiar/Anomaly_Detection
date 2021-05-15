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
        
        aux = {}
        
        print('             .Executing for granularities', gra_list)
        for gra in gra_list:
            print('\n\n             .Iter: {} - Granularity: {}'.format(n_i, gra))
            print('                 .ADP (5th Method)')
            output = ADP_Offline_Granularity_Iteration_5th(streaming_data, gra)
            aux ['granularity_'+str(gra)] = output
        ADP_outputs['iteration_'+str(n_i)] = aux

    data_analysis['ADP_outputs'] = ADP_outputs
    
    ### >>> ADP_outputs['iteration_0']['granularity_1'].keys()
    ### >>> dict_keys(['centre', 'IDX', 'Param', 'n_data_clouds'])
    
    print('\n        ====Data Processing Complete====\n' )
    print('=*='*17 ) 
    print('      ==== Commencing Data Analysis ====')
    ### Create target

    y =np.ones((len(reduced_background)+len(reduced_signal)))
    y[len(reduced_background):] = -1

    data_clouds_dic = {}
   
    print('         .Formating data_clouds_dic')

    for it in ADP_outputs:
        aux = {}
        for gra in ADP_outputs[it]:
            idx = ADP_outputs[it][gra]['IDX']
            u = np.unique(idx)
            aux2 = {}
            for i in u:
                aux3 = {}
                data = []
                target=[]

                for j in range(len(idx)):
                    if idx[j] == i:
                        data.append(list(streaming_data[j]))
                        target.append(y[j])

                aux3 ['data'] = data
                aux3 ['target'] = target
                aux2['data_cloud_'+str(i)] = aux3
            aux[gra] = aux2
        data_clouds_dic[it] = aux
            
    data_analysis['data_clouds_dic'] = data_clouds_dic
    ### >>> data_clouds_dic['iteration_0']['granularity_1']['data_cloud_0'].keys()
    ### >>> dict_keys(['data', 'target'])

    data_clouds_info = {}

    print('         .Formating data_clouds_info')

    for it in data_clouds_dic:
        aux = {}
        for gra in data_clouds_dic[it]:
            aux2 = {
                'n_data_clouds':len(data_clouds_dic[it][gra])
            }
            aux3={}
            aux4={}
            for dc in data_clouds_dic[it][gra]:
                target = data_clouds_dic[it][gra][dc]['target']
                aux3[dc] = len(target)
                aux4[dc] = target.count(-1)
            aux2['n_events_p_dc'] = aux3
            aux2['n_anom_p_dc'] = aux4
            aux[gra] = aux2
        data_clouds_info[it] = aux
    
    data_analysis['data_clouds_info'] = data_clouds_info

    ### >>> data_clouds_info['iteration_0']["granularity_1"].keys()
    ### >>> dict_keys(['n_data_clouds', 'n_events_p_dc', 'n_anom_p_dc'])

    ### >>> data_clouds_info['iteration_0']["granularity_1"]['n_events_p_dc'].keys()
    ### >>> dict_keys(['data_cloud_0', 'data_cloud_1', 
    # 'data_cloud_2', 'data_cloud_3'])

    ### >>> data_clouds_info['iteration_0']["granularity_1"]['n_anom_p_dc'].keys()\
    ### >>> dict_keys(['data_cloud_0', 'data_cloud_1', 
    # 'data_cloud_2', 'data_cloud_3'])

    with open('data_analysis.pkl', 'wb') as fp:
        pickle.dump(data_analysis, fp)

    print('\n        ====Data Analysis Complete====\n' )
    print('=*='*17 )

if __name__ == '__main__':
    main() 