import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pickle
import SODA
import threading
from datetime import datetime
from psutil import cpu_percent, swap_memory
from progress.bar import Bar
from sklearn.neighbors import kneighbors_graph 
from scipy.sparse.linalg import expm
import scipy.sparse 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import sklearn
from sklearn.utils.validation import check_array
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model

import ADP

from matplotlib.backends.backend_pdf import PdfPages

class performance(threading.Thread):
    # Declares variables for perfomance analysis:
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.control = True
    
    def run(self):
        cpu_p = []
        ram_p = []
        ram_u = []
        while self.control:
            cpu_p.append(cpu_percent(interval=1, percpu=True))
            ram_p.append(swap_memory().percent)
            ram_u.append(swap_memory().used/(1024**3))
        self.mean_cpu_p = np.mean(cpu_p)
        self.mean_ram_p = np.mean(ram_p)
        self.mean_ram_u = np.mean(ram_u)
        self.max_cpu_p = np.max(np.mean(cpu_p, axis=1))
        self.max_ram_p = np.max(ram_p)
        self.max_ram_u = np.max(ram_u)
    
    def stop(self):
        self.control = False
    
    def join(self):
        threading.Thread.join(self)
        out = {'mean_cpu_p': self.mean_cpu_p,
               'mean_ram_p': self.mean_ram_p,
               'mean_ram_u': self.mean_ram_u,
               'max_cpu_p': self.max_cpu_p,
               'max_ram_p': self.max_ram_p,
               'max_ram_u': self.max_ram_u}
        return out

def divide(data, n_windows = 100, n_samples = 50):  
    """Divide the data in n_samples, and the samples are equaly distributed in n_windows
    -- Input
    - data = data to split
    - n_windows = number of windows to separete the data
    - n_samples = number of samples of the output
    -- Output
    - reduced_data = splited data with n_samples
    - data_sample_id = id of the splited data"""  
    L, W = data.shape
    
    # Checking if the windows can be of the same size 
    
    if int(L % n_windows) != 0:
        
        # Checking if we need to pick the same amount of data from each window
        
        if int(n_samples % n_windows) != 0 or (n_samples/n_windows) % 1 != 0:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window + int(n_windows*((n_samples/n_windows) % 1))):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample
            
        # Even amount of data 
            
        else:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample 
                        
    # Windows of same size 

    else:
        
        # Checking if we need to pick the same amount of data from each window
        
        if int(n_samples % n_windows) != 0 or (n_samples/n_windows) % 1 != 0:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))
            
            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window + int(n_windows*((n_samples/n_windows) % 1))):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample
            
        # Even amount of data
            
        else:
        
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                for j in range(samples_per_window):
                    sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window))
                    new_line = data[sample]
                    reduced_data[j+(i*samples_per_window)] = new_line
                    data_sample_id[j+(i*samples_per_window)] = sample
        
    return reduced_data, data_sample_id

def Normalisation(background_train,streaming_data):
    """Use standart deviation to normalise the data
    -- Input
    - data
    -- Output
    - Normalised data
    """

    scaler = StandardScaler().fit(background_train)
    background_train = scaler.transform(background_train)

    # Normalizing whole data
    scaler = StandardScaler().fit(streaming_data)
    streaming_data = scaler.transform(streaming_data)

    return background_train,streaming_data

def PCA_Analysis(mantained_variation, attributes_influence,laplace=True):
    """Create and save the PCA model
    -- Input
    - mantained_variation = variation mantained for each PC
    - attributes_influence = influence of each attribute on the model 
    -- Output
    - Saves plot figures in results folder
    """

    # Plots the variation mantained by each PC

    fig = plt.figure(figsize=[16,8])
    ax = fig.subplots(1,1)
    ax.bar(x=['PC' + str(x) for x in range(1,(len(mantained_variation)+1))],height=mantained_variation)

    ax.set_ylabel('Percentage of Variance Held',fontsize=20)
    ax.set_xlabel('Principal Components',fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid()

    #fig.savefig('/AtlasDisk/user/pestana/Output/results/Percentage_of_Variance_Held.png', bbox_inches='tight')
    fig.savefig('results/Percentage_of_Variance_Held.png', bbox_inches='tight')

                        
    sorted_sensors_contribution = attributes_influence.values[:]      
                        
    # Ploting Cntribution Attributes influence
                        
    fig = plt.figure(figsize=[25,8])

    fig.suptitle('Attributes Weighted Contribution Percentage', fontsize=16)

    ax = fig.subplots(1,1)

    sorted_sensors_contribution = sorted_sensors_contribution.ravel()
    ax.bar(x=list(attributes_influence.columns),height=sorted_sensors_contribution)
    plt.ylabel('Relevance Percentage',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    
    #fig.savefig('/AtlasDisk/user/pestana/Output/results/Attributes_Contribution.png', bbox_inches='tight')
    fig.savefig('results/Attributes_Contribution.png', bbox_inches='tight')

    return

def PCA_Projection(background_train,streaming_data, N_PCs, maintained_features=0):
    """Transform Data with PCA and normalize
    -- Input
    - Offline data
    - Streaming data
    - N_PCs = number of PCs to calculate
    -- Output
    - Projected Offline data
    - Projected Streaming data
    - Variation Mantained
    - Attributes Influence
    """

    """
    data = np.vstack((background_train,streaming_data))

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)

    # Calcules the PCA and projects the data-set into them
    
    pca= PCA(n_components = N_PCs)
    pca.fit(data)


    proj_background_train = pca.transform(background_train)
    proj_streaming_data = pca.transform(streaming_data)

    np.savetxt('proj_background_train_junto_norm.csv',proj_background_train,delimiter=',')
    np.savetxt('proj_streaming_data_junto_norm.csv',proj_streaming_data,delimiter=',')
    """
    L, W = background_train.shape

    scaler = StandardScaler().fit(background_train)
    norm_background_train = scaler.transform(background_train)
    norm_streaming_data = scaler.transform(streaming_data)

    pca = PCA(n_components = N_PCs)
    pca.fit(norm_background_train)
    proj_background_train = pca.transform(norm_background_train)
    proj_streaming_data = pca.transform(norm_streaming_data)

    np.savetxt('proj_background_train_sep_std.csv',proj_background_train,delimiter=',')
    np.savetxt('proj_streaming_data_sep_std.csv',proj_streaming_data,delimiter=',')

    # Calculates the total variance maintained by each PCs
            
    pca_variation = pca.explained_variance_ratio_ * 100
    
    print('             .Normal Variation maintained: %.2f' % np.round(pca_variation.sum(), decimals = 2), file=open("log_file.txt", "a"))
 

    ### Attributes analyses ###
    columns = ['C{}'.format(i) for i in range(W)] 
    '''columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A',
           'min_v', 'max_v', 'mean', 'var', 'skew', 'kurt', 
           'moment2', 'moment3', 'moment4', 
           'moment5', 'moment6', 'moment7', 'moment8', 'moment9', 'moment10',
           'cumulant1', 'cumulant2', 'cumulant3', 'cumulant4','kstatar1', 'kstatvar2']'''

    # Gets eigen vectors information from the trained pca object
    eigen_matrix = np.array(pca.components_)

    # Inverting negative signals
    eigen_matrix = pow((pow(eigen_matrix,2)),0.5) 

    # Calculates the feature contribution

    for i in range (eigen_matrix.shape[0]):
        LineSum = sum(eigen_matrix[i,:])
        for j in range (eigen_matrix.shape[1]):
            eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)

    weighted_contribution = np.zeros((eigen_matrix.shape[1]))

    for i in range (eigen_matrix.shape[1]):
        NumeratorSum = 0
        for j in range (N_PCs):
            NumeratorSum += eigen_matrix[j,i] * pca_variation[j]

        weighted_contribution[i] = NumeratorSum / sum(pca_variation)

    weighted_contribution = weighted_contribution.reshape((1,-1))

    # Sorting attributes by their contribution values 
                        
    attributes_contribution = pd.DataFrame (weighted_contribution, columns = columns)
                        
    attributes_contribution = attributes_contribution.sort_values(by=0, axis=1,ascending=False)

    return proj_background_train, proj_streaming_data, pca_variation, attributes_contribution

def statistics_attributes(data):
    """
    When xyz_attributes=True:
       Concatenate with the data, statistics attributes for each event.
       Currently Applied:
        - Minimum Value
        - Maximum Value
        - Mean
        - Variance
        - Skewness
        - Kurtosis
        - 2nd Central Moment
        - 3rd Central Moment
        - 4th Central Moment
        - Bayesian Confidence Interval (Min and Max)
        More attributes may be added later
       
       -- Input
       - data [numpy.array]
       -- Output
       - output_data = data with adition of statistical features for each line [numpy.array]
       """

    L, W = data.shape

    _, (min_v, max_v), mean, var, skew, kurt = stats.describe(data.transpose())
    
    # Minimum Value
    min_v = min_v.reshape(-1,1)
    
    min_v = check_array(min_v) # Minimum Value

    # Maximum Value
    max_v = max_v.reshape(-1,1)

    max_v = check_array(max_v) # Maximum Value

    # Mean
    mean = mean.reshape(-1,1)
    
    mean = check_array(mean) # Mean

    # Variance
    var = var.reshape(-1,1) 
    
    var = check_array(var) # Variance

    # Skewness
    skew = skew.reshape(-1,1)
    
    skew = check_array(skew) # Skewness

    # Kurtosis
    kurt = kurt.reshape(-1,1)
    
    kurt = check_array(kurt) # Kurtosis

    # 2nd Central Moment
    moment2 = stats.moment(data.transpose(), moment=2).reshape(-1,1)
    
    moment2 = check_array(moment2) # 2nd Central Moment

    # 3rd Central Moment
    moment3 = stats.moment(data.transpose(), moment=3).reshape(-1,1)
    
    moment3 = check_array(moment3) # 3rd Central Moment

    # 4th Central Moment
    moment4 = stats.moment(data.transpose(), moment=4).reshape(-1,1)
    
    moment4 = check_array(moment4) # 4th Central Moment
  
    output_data = np.concatenate((data, min_v, max_v, mean, var, skew, kurt, moment2, moment3, moment4), axis=1)
                                        
    return output_data

def SODA_Granularity_Iteration(offline_data,streaming_data,gra,n_backgound,Iteration):
    # Formmating  Data
    offline_data = np.matrix(offline_data)
    
    L1 = len(offline_data)

    streaming_data = np.matrix(streaming_data)
    
    data = np.concatenate((offline_data, streaming_data), axis=0)

    # Dreate data frames to save each iteration result.

    detection_info = pd.DataFrame(np.zeros((1,7)).reshape((1,-1)), columns=['Granularity',
                                                                            'True_Positive', 'True_Negative',
                                                                            'False_Positive','False_Negative', 
                                                                            'N_Groups', 'Time_Elapsed'])

    begin = datetime.now()

    detection_info.loc[0,'Granularity'] = gra
    
    Input = {'GridSize':gra, 'StaticData':offline_data, 'DistanceType': 'euclidean'}

    out = SODA.SelfOrganisedDirectionAwareDataPartitioning(Input,'Offline')

    # Concatanating IDs and creating labels
    
    label = np.zeros((len(streaming_data)))
    label[n_backgound:] = 1

    decision = np.zeros((len(streaming_data)))
    
    Input['StreamingData'] = streaming_data
    Input['SystemParams'] = out['SystemParams']
    Input['AllData'] = data

    online_out = SODA.SelfOrganisedDirectionAwareDataPartitioning(Input,'Evolving')

    signal_centers = online_out['C']
    soda_labels = online_out['IDX']
    online_soda_labels = soda_labels[(L1):]

    cloud_info = pd.DataFrame(np.zeros((len(signal_centers),4)),columns=['Total_Samples','Old_Samples','Percentage_Old_Samples', 'Percentage_of_Samples'])
    
    for j in range (len(soda_labels)):
        if j < L1:
            cloud_info.loc[int(soda_labels[j]),'Old_Samples'] += 1
        cloud_info.loc[int(soda_labels[j]),'Total_Samples'] += 1

    cloud_info.loc[:,'Percentage_Old_Samples'] = cloud_info.loc[:,'Old_Samples'] * 100 / cloud_info.loc[:,'Total_Samples']
    cloud_info.loc[:,'Percentage_of_Samples'] = cloud_info.loc[:,'Total_Samples'] * 100/ cloud_info.loc[:,'Total_Samples'].sum()

    anomaly_clouds=[]
    n_anomalies = 0

    for j in range(len(signal_centers)):
        if cloud_info.loc[j,'Percentage_Old_Samples'] == 0 :
            n_anomalies += cloud_info.loc[j,'Total_Samples']
            anomaly_clouds.append(j)
    
    if n_anomalies != 0:
        for j in range(len(online_soda_labels)): 
            if online_soda_labels[j] in anomaly_clouds:
                decision[j] = 1
        
    for j in range(len(label)):
        if label[j] == 1:
            if decision[j] == label[j]:
                detection_info.loc[0,'True_Positive'] += 1
            
            else:
                detection_info.loc[0,'False_Negative'] += 1
                
        else:
            if decision[j] == label[j]:
                detection_info.loc[0,'True_Negative'] += 1
            
            else:
                detection_info.loc[0,'False_Positive'] += 1
    
    detection_info.loc[0,'N_Groups'] = max(soda_labels)+1

    final = datetime.now()
    detection_info.loc[0,'Time_Elapsed'] = (final - begin)
    print(detection_info.to_string(index=False))
    detection_info.to_csv('results/SODA_detection_info_{}_{}.csv'.format(gra,Iteration), index=False)
    
def calc_eos(data):
    ### Parametros comuns para todos os calculos ###
    L, W = data.shape
    
    # Extrair a media da entrada
    mean = data.mean(axis=1).reshape(-1,1)
    dataEnt = data - mean
    
    ### Criar matriz de saída
    W2 = int(np.ceil(W/2))
    dataSaida = np.zeros((L,3*W+3*W2))
    
    ### Produto Vetorial para EOS2
    EOS2 = (dataEnt**2).sum(axis=1)
    
    ### calc_mod ###
    for j in range(W):
        eos2_mod = np.zeros(L) # eos2_mod
        eos3_mod = np.zeros(L) # eos3_mod
        eos4_mod1 = np.zeros(L) # eos4_mod
        eos4_mod2 = np.zeros(L) # eos4_mod
        for i in range(W):
            eos2_mod += dataEnt[:,i]*dataEnt[:,(i+j)%W] # eos2_mod
            eos3_mod += dataEnt[:,i]*dataEnt[:,i]*dataEnt[:,(i+j)%W] # eos3_mod
            eos4_mod1 += dataEnt[:,i]*dataEnt[:,(i+j)%W]**3 # eos4_mod
            eos4_mod2 += dataEnt[:,i]*dataEnt[:,(i+j)%W] # eos4_mod
            
        dataSaida[:,j] = eos2_mod/W # eos2_mod
        dataSaida[:,j+W+W2] = eos3_mod/W # eos3_mod
        dataSaida[:,j+2*W+2*W2] = eos4_mod1/W - (3/W**2)*eos4_mod2*EOS2 # eos4_mod
    
    ### calc_norm
    for j in range(W2):
        eos2_norm = np.zeros(L) # eos2_norm
        eos3_norm = np.zeros(L) # eos3_norm
        eos4_norm1 = np.zeros(L) # eos4_norm
        eos4_norm2 = np.zeros(L) # eos4_norm
        for i in range(W2):
            eos2_norm += dataEnt[:,i]*dataEnt[:,i+j] # eos2_norm
            eos3_norm += dataEnt[:,i]*dataEnt[:,i+j]*dataEnt[:,i+j] # eos3_norm
            eos4_norm1 += dataEnt[:,i]*dataEnt[:,i+j]**3 # eos4_norm
            eos4_norm2 += dataEnt[:,i]*dataEnt[:,i+j] # eos4_norm
            
        dataSaida[:,j+W] = eos2_norm/W2 # eos2_norm
        dataSaida[:,j+2*W+W2] = eos3_norm/W2 # eos3_norm
        dataSaida[:,j+3*W+2*W2] = eos4_norm1/W2 - (3/W2**2)*eos4_norm2*EOS2 # eos4_norm
    
    return dataSaida

def autoencoder(static, streaming, reduction=9, epochs=200, batch_size=32):
    # Normalize Features
    norm = Normalizer(norm='max').fit(static.T)
    static = norm.transform(static.T).T
    streaming = norm.transform(streaming.T).T

    L, W = static.shape
    
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

    output = Dense(W, activation='linear')(d)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(static, static, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(streaming,streaming))
    encoder = Model(inputs=visible, outputs=bottleneck)

    static_encoded = encoder.predict(static)
    streaming_encoded = encoder.predict(streaming)

    return static_encoded, streaming_encoded

############################################################################################
# ---------------------------------------------------------------------------------------- #
# ------------------------------------ NEW FUNCTIONS ------------------------------------- #
# ---------------------------------------------------------------------------------------- #
############################################################################################
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def clouds_stats(data):
    data = np.array(data)
    data = data[:,:-1]
    L, W = data.shape

    _, (min_v, max_v), mean, var, skew, kurt = stats.describe(data)
    # 2nd Central Moment
    moment2 = stats.moment(data, moment=2)

    # 3rd Central Moment
    moment3 = stats.moment(data, moment=3)

    # 4th Central Moment
    moment4 = stats.moment(data, moment=4)
  
    output_data = np.concatenate((min_v, max_v, mean, var, skew, kurt, moment2, moment3, moment4), axis=0)

    output_data[np.isnan(output_data)] = 0
    return output_data

def HOS_clouds_stats_tau(data, tau=0):
    ### Parametros comuns para todos os calculos ###
    data = np.array(data)
    data = data[:,:-1]
    L, W = data.shape
    
    # Extrair a media da entrada
    mean = data.mean(axis=0)
    dataEnt = data - mean
    
    ### Produto Vetorial para EOS2
    EOS2 = (dataEnt**2).sum(axis=0)
    
    ### Criar matriz de saída
    L2 = int(np.ceil(L/2))
    dataSaida = np.zeros(6*W)

    eos2_mod = 0 # eos2_mod
    eos3_mod = 0 # eos3_mod
    eos4_mod1 = 0 # eos4_mod
    eos4_mod2 = 0 # eos4_mod
    for i in range(L):
        eos2_mod += dataEnt[i,:]*dataEnt[(i+tau)%L,:] # eos2_mod
        eos3_mod += dataEnt[i,:]*dataEnt[i,:]*dataEnt[(i+tau)%L,:] # eos3_mod
        eos4_mod1 += dataEnt[i,:]*dataEnt[(i+tau)%L,:]**3 # eos4_mod
        eos4_mod2 += dataEnt[i,:]*dataEnt[(i+tau)%L,:] # eos4_mod
    
    dataSaida[0:W] = eos2_mod/L # eos2_mod
    dataSaida[W:2*W] = eos3_mod/L # eos3_mod
    dataSaida[2*W:3*W] = eos4_mod1/L - (3/L**2)*eos4_mod2*EOS2 # eos4_mod
        
    eos2_norm = 0 # eos2_norm
    eos3_norm = 0 # eos3_norm
    eos4_norm1 = 0 # eos4_norm
    eos4_norm2 = 0 # eos4_norm
    for i in range(L2):
        eos2_norm += dataEnt[i,:]*dataEnt[i+tau,:] # eos2_norm
        eos3_norm += dataEnt[i,:]*dataEnt[i+tau,:]*dataEnt[i+tau,:] # eos3_norm
        eos4_norm1 += dataEnt[i,:]*dataEnt[i+tau,:]**3 # eos4_norm
        eos4_norm2 += dataEnt[i,:]*dataEnt[i+tau,:] # eos4_norm
            
    dataSaida[3*W:4*W] = eos2_norm/L2 # eos2_norm
    dataSaida[4*W:5*W] = eos3_norm/L2 # eos3_norm
    dataSaida[5*W:6*W] = eos4_norm1/L2 - (3/L2**2)*eos4_norm2*EOS2 # eos4_norm
    
    return dataSaida

def PCA_clouds(static, gra):
    L, W = static.shape

    scaler = StandardScaler().fit(static)
    scaled_static = scaler.transform(static)

    i = 1
    pca_var = 0
    while pca_var < 70 and (i <= 2*W/3) and (i<L):
        pca = PCA(n_components = i)
        pca.fit(scaled_static)
        proj_static = pca.transform(scaled_static)
        i += 1

        # Calculates the total variance maintained by each PCs            
        pca_variation = pca.explained_variance_ratio_ * 100
        pca_var = pca_variation.sum()

    N_PCs = i-1
    print('             .Normal Variation maintained: %.2f' % np.round(pca_variation.sum(), decimals = 2))
    print('             .Principal Components Kept:', N_PCs)
 

    ### Attributes analyses ###
    columns1 = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A']

    columns2 = ['2_mod', '3_mod', '4_mod', '2_norm', '3_norm', '4_norm']
    '''columns2 = ['min_v', 'max_v', 'mean', 'var', 'skew', 'kurt', 
                            'moment2', 'moment3', 'moment4']'''


    columns = []
    for c2 in columns2:
        for c1 in columns1:
            columns.append('{}__{}'.format(c1,c2))

    # Gets eigen vectors information from the trained pca object
    eigen_matrix = np.array(pca.components_)

    # Inverting negative signals
    eigen_matrix = pow((pow(eigen_matrix,2)),0.5) 

    # Calculates the feature contribution
    for i in range (eigen_matrix.shape[0]):
        LineSum = sum(eigen_matrix[i,:])
        for j in range (eigen_matrix.shape[1]):
            eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)

    weighted_contribution = np.zeros((eigen_matrix.shape[1]))
    for i in range (eigen_matrix.shape[1]):
        NumeratorSum = 0
        for j in range (N_PCs):
            NumeratorSum += eigen_matrix[j,i] * pca_variation[j]
        weighted_contribution[i] = NumeratorSum / sum(pca_variation)
    weighted_contribution = weighted_contribution.reshape((1,-1))

    # Sorting attributes by their contribution values   
    attributes_contribution = pd.DataFrame (weighted_contribution, columns = columns)
                        
    attributes_contribution = attributes_contribution.sort_values(by=0, axis=1,ascending=False)
    
    # Plots the variation mantained by each PC
    fig = plt.figure(figsize=[16,8])
    ax = fig.subplots(1,1)
    ax.bar(x=['PC' + str(x) for x in range(1,(len(pca_variation)+1))],height=pca_variation)
    ax.set_ylabel('Percentage of Variance Held',fontsize=20)
    ax.set_xlabel('Principal Components',fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid()
    fig.savefig('results/Percentage_of_Variance_Held_{}.png'.format(gra), bbox_inches='tight')
    plt.close()

                        
    sorted_sensors_contribution = attributes_contribution.values[:]      
                        
    # Ploting Cntribution Attributes influence
    fig = plt.figure(figsize=[25,8])
    fig.suptitle('Attributes Weighted Contribution Percentage', fontsize=16)
    ax = fig.subplots(1,1)
    sorted_sensors_contribution = sorted_sensors_contribution.ravel()
    ax.bar(x=list(attributes_contribution.columns),height=sorted_sensors_contribution)
    plt.ylabel('Relevance Percentage',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    fig.savefig('results/Attributes_Contribution_{}.png'.format(gra), bbox_inches='tight')
    plt.close()   
                        
    # Ploting Cntribution Attributes influence
    fig = plt.figure(figsize=[25,8])
    fig.suptitle('20th Attributes Weighted Contribution Percentage', fontsize=16)
    ax = fig.subplots(1,1)
    sorted_sensors_contribution = sorted_sensors_contribution.ravel()
    ax.bar(x=list(attributes_contribution.columns)[:20],height=sorted_sensors_contribution[:20])
    plt.ylabel('Relevance Percentage',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    fig.savefig('results/20th_Attributes_Contribution_{}.png'.format(gra), bbox_inches='tight')
    plt.close()


    return scaler, pca, proj_static

def NEW_ADPOffline_Granularity_Iteration_first(static, streaming, gra, b_test, n_i):
    begin = datetime.now()

    L1, W = static.shape
    L2, _ = streaming.shape

    ##################################
    ##### ----- STATIC ADP ----- #####
    ##### ---------------------- #####

    Input = {'data': static,
             'granularity': gra,
             'distancetype': 'euclidean'}
            
    static_output = ADP.ADP(Input, 'Offline')

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####
            
    IDX = static_output['IDX']
    static_n_centers = max(IDX) + 1

    labeled_static = np.concatenate((static, IDX.reshape(-1,1)), axis=1)

    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    static_df = pd.DataFrame(labeled_static, columns=columns)
    static_cloud_info = []
    for i in range(static_n_centers):
        static_cloud_info.append(HOS_clouds_stats_tau(static_df[static_df['LABEL'] == i]))
    static_clouds = np.array(static_cloud_info)
    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    scaler, pca, proj_static = PCA_clouds(static_clouds, gra)

    #########################################
    ##### ----- TRAIN CLASSIFIERS ----- #####
    ##### ----------------------------- #####
    clf_forest = IsolationForest(contamination=0, bootstrap=True, random_state=0)
    clf_forest.fit(proj_static)

    clf_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_svm.fit(proj_static)

    clf_LOF = LocalOutlierFactor(novelty = True)
    clf_LOF.fit(proj_static)

    clf_elliptic = EllipticEnvelope(contamination=0, random_state=0)
    clf_elliptic.fit(proj_static)
    #####################################
    ##### ----- STREAMING ADP ----- #####
    ##### ------------------------- #####
    data = np.concatenate((static,streaming), axis=0)

    # Dreate data frames to save each iteration result.
    detection_info = pd.DataFrame(np.zeros((5,8)), columns=['Method', 'Granularity',
                                                            'True_Positive', 'True_Negative',
                                                            'False_Positive','False_Negative', 
                                                            'N_Groups', 'Time_Elapsed'])

    detection_info.loc[0,'Method'] = 'Old'
    detection_info.loc[1,'Method'] = 'IsolationForest'
    detection_info.loc[2,'Method'] = 'SVM-rbf'
    detection_info.loc[3,'Method'] = 'LOF'
    detection_info.loc[4,'Method'] = 'EllipticEnvelope'

    detection_info['Granularity'] = gra

    # Concatanating IDs and creating labels
    label = np.zeros((L2))
    label[b_test:] = 1
    decision = np.zeros((L2))

    # Execute ADP for Streaming data
    Input = {'data': data,
             'granularity': gra,
             'distancetype': 'euclidean'}
    output = ADP.ADP(Input, 'Offline')

    on_center = output['centre']
    on_IDX = output['IDX']
    online_labels = output['IDX'][L1:]

    # Detect Anomalies Based on Old Samples in Cloud
    cloud_info = pd.DataFrame(np.zeros((len(on_center),4)),columns=['Total_Samples','Old_Samples',
                                                                    'Percentage_Old_Samples', 'Percentage_of_Samples'])
    
    for j in range (len(on_IDX)):
        if j < L1:
            cloud_info.loc[int(on_IDX[j]),'Old_Samples'] += 1
        cloud_info.loc[int(on_IDX[j]),'Total_Samples'] += 1

    cloud_info.loc[:,'Percentage_Old_Samples'] = cloud_info.loc[:,'Old_Samples'] * 100 / cloud_info.loc[:,'Total_Samples']
    cloud_info.loc[:,'Percentage_of_Samples'] = cloud_info.loc[:,'Total_Samples'] * 100/ cloud_info.loc[:,'Total_Samples'].sum()

    anomaly_clouds=[]
    n_anomalies = 0

    for j in range(len(on_center)):
        if cloud_info.loc[j,'Percentage_Old_Samples'] == 0 :
            n_anomalies += cloud_info.loc[j,'Total_Samples']
            anomaly_clouds.append(j)
    
    if n_anomalies != 0:
        for j in range(len(online_labels)): 
            if online_labels[j] in anomaly_clouds:
                decision[j] = 1
        
    for j in range(len(label)):
        if label[j] == 1:
            if decision[j] == label[j]:
                detection_info.loc[0,'True_Positive'] += 1
            else:
                detection_info.loc[0,'False_Negative'] += 1     
        else:
            if decision[j] == label[j]:
                detection_info.loc[0,'True_Negative'] += 1
            else:
                detection_info.loc[0,'False_Positive'] += 1
    
    detection_info.loc[0,'N_Groups'] = max(on_IDX) + 1

    final = datetime.now()
    detection_info.loc[0,'Time_Elapsed'] = (final - begin)

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####

    labeled_streaming = np.concatenate((streaming, online_labels.reshape(-1,1)), axis=1)
    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    streaming_df = pd.DataFrame(labeled_streaming, columns=columns)

    labels_df = pd.DataFrame(np.zeros((len(anomaly_clouds),3)),columns=['Cloud_Id','Background_Samples','Signal_Samples'])
    labels_df['Cloud_Id'] = anomaly_clouds
    for i in range(L2):
        if online_labels[i] in anomaly_clouds:
            idx = anomaly_clouds.index(online_labels[i])
            if label[i] == 0:
                labels_df.loc[idx,'Background_Samples'] += 1
            if label[i] == 1:
                labels_df.loc[idx,'Signal_Samples'] += 1
    labels_df.to_csv('results/Online_Data_Clouds_{}_{}.csv'.format(gra,n_i), index=False)

    streaming_cloud_info = []
    for i in anomaly_clouds:
        streaming_cloud_info.append(HOS_clouds_stats_tau(streaming_df[streaming_df['LABEL'] == i]))
    streaming_clouds = np.array(streaming_cloud_info)

    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    if n_anomalies != 0:
        if len(anomaly_clouds) > 1:
            scaled_streaming_clouds = scaler.transform(streaming_clouds)
            proj_streaming = pca.transform(scaled_streaming_clouds)
        else:
            scaled_streaming_clouds = scaler.transform(streaming_clouds.reshape(1, -1))
            proj_streaming = pca.transform(scaled_streaming_clouds.reshape(1, -1))

        metade = datetime.now()
        #############################################
        ##### ----- DETECTION INFO FOREST ----- #####
        ##### --------------------------------- #####
        y_pred_forest_static = clf_forest.predict(proj_static)
        y_pred_forest = clf_forest.predict(proj_streaming)

        att_anomaly_clouds = []
        for i in range(len(anomaly_clouds)):
            if y_pred_forest[i] == 1:
                att_anomaly_clouds.append(anomaly_clouds[i])

        decision = np.zeros((L2))
        for j in range(len(online_labels)): 
                if online_labels[j] in att_anomaly_clouds:
                    decision[j] = 1
            
        for j in range(len(label)):
            if label[j] == 1:
                if decision[j] == label[j]:
                    detection_info.loc[1,'True_Positive'] += 1
                else:
                    detection_info.loc[1,'False_Negative'] += 1     
            else:
                if decision[j] == label[j]:
                    detection_info.loc[1,'True_Negative'] += 1
                else:
                    detection_info.loc[1,'False_Positive'] += 1
        detection_info.loc[1,'N_Groups'] = max(on_IDX) + 1

        final = datetime.now()
        detection_info.loc[1,'Time_Elapsed'] = (final - begin)
        ##########################################
        ##### ----- DETECTION INFO SVM ----- #####
        ##### ------------------------------ #####
        y_pred_svm_static = clf_svm.predict(proj_static)
        y_pred_svm = clf_svm.predict(proj_streaming)

        att_anomaly_clouds = []
        for i in range(len(anomaly_clouds)):
            if y_pred_svm[i] == 1:
                att_anomaly_clouds.append(anomaly_clouds[i])

        decision = np.zeros((L2))
        for j in range(len(online_labels)): 
                if online_labels[j] in att_anomaly_clouds:
                    decision[j] = 1
            
        for j in range(len(label)):
            if label[j] == 1:
                if decision[j] == label[j]:
                    detection_info.loc[2,'True_Positive'] += 1
                else:
                    detection_info.loc[2,'False_Negative'] += 1     
            else:
                if decision[j] == label[j]:
                    detection_info.loc[2,'True_Negative'] += 1
                else:
                    detection_info.loc[2,'False_Positive'] += 1
        detection_info.loc[2,'N_Groups'] = max(on_IDX) + 1
        final2 = datetime.now()
        detection_info.loc[2,'Time_Elapsed'] = (metade - begin) + (final2-final)

        ##########################################
        ##### ----- DETECTION INFO LOF ----- #####
        ##### ------------------------------ #####
        y_pred_LOF_static = clf_LOF.predict(proj_static)
        y_pred_LOF = clf_LOF.predict(proj_streaming)

        att_anomaly_clouds = []
        for i in range(len(anomaly_clouds)):
            if y_pred_LOF[i] == 1:
                att_anomaly_clouds.append(anomaly_clouds[i])

        decision = np.zeros((L2))
        for j in range(len(online_labels)): 
                if online_labels[j] in att_anomaly_clouds:
                    decision[j] = 1
            
        for j in range(len(label)):
            if label[j] == 1:
                if decision[j] == label[j]:
                    detection_info.loc[3,'True_Positive'] += 1
                else:
                    detection_info.loc[3,'False_Negative'] += 1     
            else:
                if decision[j] == label[j]:
                    detection_info.loc[3,'True_Negative'] += 1
                else:
                    detection_info.loc[3,'False_Positive'] += 1
        detection_info.loc[3,'N_Groups'] = max(on_IDX) + 1
        final = datetime.now()
        detection_info.loc[3,'Time_Elapsed'] = (metade - begin) + (final-final2)
        ###############################################
        ##### ----- DETECTION INFO Elliptic ----- #####
        ##### ----------------------------------- #####
        y_pred_elliptic_static = clf_elliptic.predict(proj_static)
        y_pred_elliptic = clf_elliptic.predict(proj_streaming)

        att_anomaly_clouds = []
        for i in range(len(anomaly_clouds)):
            if y_pred_elliptic[i] == 1:
                att_anomaly_clouds.append(anomaly_clouds[i])

        decision = np.zeros((L2))
        for j in range(len(online_labels)): 
                if online_labels[j] in att_anomaly_clouds:
                    decision[j] = 1
            
        for j in range(len(label)):
            if label[j] == 1:
                if decision[j] == label[j]:
                    detection_info.loc[4,'True_Positive'] += 1
                else:
                    detection_info.loc[4,'False_Negative'] += 1     
            else:
                if decision[j] == label[j]:
                    detection_info.loc[4,'True_Negative'] += 1
                else:
                    detection_info.loc[4,'False_Positive'] += 1
        detection_info.loc[4,'N_Groups'] = max(on_IDX) + 1
        final2 = datetime.now()
        detection_info.loc[4,'Time_Elapsed'] = (metade - begin) + (final2-final)


    print("\n\n.Detection Info")
    print(detection_info.to_string(index=False))
    detection_info.to_csv('results/First_ADP_detection_info_{}_{}.csv'.format(gra,n_i), index=False)

def NEW_ADPOffline_Granularity_Iteration_second(static, streaming, gra, b_test, n_i):
    begin = datetime.now()

    L1, W = static.shape
    L2, _ = streaming.shape

    ##################################
    ##### ----- STATIC ADP ----- #####
    ##### ---------------------- #####

    Input = {'data': static,
             'granularity': gra,
             'distancetype': 'euclidean'}
            
    static_output = ADP.ADP(Input, 'Offline')

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####
            
    IDX = static_output['IDX']
    static_n_centers = max(IDX) + 1

    labeled_static = np.concatenate((static, IDX.reshape(-1,1)), axis=1)

    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    static_df = pd.DataFrame(labeled_static, columns=columns)
    static_cloud_info = []
    for i in range(static_n_centers):
        static_cloud_info.append(HOS_clouds_stats_tau(static_df[static_df['LABEL'] == i]))
    static_clouds = np.array(static_cloud_info)
    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    scaler, pca, proj_static = PCA_clouds(static_clouds, gra)

    #########################################
    ##### ----- TRAIN CLASSIFIERS ----- #####
    ##### ----------------------------- #####
    clf_forest = IsolationForest(contamination=0, bootstrap=True, random_state=0)
    clf_forest.fit(proj_static)

    clf_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_svm.fit(proj_static)

    clf_LOF = LocalOutlierFactor(novelty = True)
    clf_LOF.fit(proj_static)

    clf_elliptic = EllipticEnvelope(contamination=0, random_state=0)
    clf_elliptic.fit(proj_static)
    #####################################
    ##### ----- STREAMING ADP ----- #####
    ##### ------------------------- #####
    data = np.concatenate((static,streaming), axis=0)
    L,_ = data.shape

    # Dreate data frames to save each iteration result.
    detection_info = pd.DataFrame(np.zeros((4,8)), columns=['Method', 'Granularity',
                                                            'True_Positive', 'True_Negative',
                                                            'False_Positive','False_Negative', 
                                                            'N_Groups', 'Time_Elapsed'])

    detection_info.loc[0,'Method'] = 'IsolationForest'
    detection_info.loc[1,'Method'] = 'SVM-rbf'
    detection_info.loc[2,'Method'] = 'LOF'
    detection_info.loc[3,'Method'] = 'EllipticEnvelope'

    detection_info['Granularity'] = gra

    # Concatanating IDs and creating labels
    label = np.zeros((L2))
    label[b_test:] = 1
    decision = np.zeros((L2))

    # Execute ADP for Streaming data
    Input = {'data': data,
             'granularity': gra,
             'distancetype': 'euclidean'}
    output = ADP.ADP(Input, 'Offline')

    on_center = output['centre']
    on_IDX = output['IDX']
    online_labels = output['IDX'][L1:]

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####

    labeled_streaming = np.concatenate((streaming, online_labels.reshape(-1,1)), axis=1)
    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    streaming_df = pd.DataFrame(labeled_streaming, columns=columns)

    streaming_cloud_info = []
    for i in np.unique(online_labels):
        streaming_cloud_info.append(HOS_clouds_stats_tau(streaming_df[streaming_df['LABEL'] == i]))
    streaming_clouds = np.array(streaming_cloud_info)
    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    if True:
        scaled_streaming_clouds = scaler.transform(streaming_clouds)
        proj_streaming = pca.transform(scaled_streaming_clouds)
        metade = datetime.now()
        #############################################
        ##### ----- DETECTION INFO FOREST ----- #####
        ##### --------------------------------- #####
        y_pred_forest = clf_forest.predict(proj_streaming)

        for i,label_idx in enumerate(np.unique(online_labels)):
            if y_pred_forest[i] == 1:
                detection_info.loc[0,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[0,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
            else:
                detection_info.loc[0,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[0,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()

        detection_info.loc[0,'N_Groups'] = max(on_IDX) + 1

        final = datetime.now()
        detection_info.loc[0,'Time_Elapsed'] = (final - begin)
        ##########################################
        ##### ----- DETECTION INFO SVM ----- #####
        ##### ------------------------------ #####
        y_pred_svm = clf_svm.predict(proj_streaming)

        for i,label_idx in enumerate(np.unique(online_labels)):
            if y_pred_svm[i] == 1:
                detection_info.loc[1,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[1,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
            else:
                detection_info.loc[1,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[1,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
                
        detection_info.loc[1,'N_Groups'] = max(on_IDX) + 1

        final2 = datetime.now()
        detection_info.loc[1,'Time_Elapsed'] = (metade - begin) + (final2-final)

        ##########################################
        ##### ----- DETECTION INFO LOF ----- #####
        ##### ------------------------------ #####
        y_pred_LOF = clf_LOF.predict(proj_streaming)

        for i,label_idx in enumerate(np.unique(online_labels)):
            if y_pred_LOF[i] == 1:
                detection_info.loc[2,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[2,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
            else:
                detection_info.loc[2,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[2,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
                
        detection_info.loc[2,'N_Groups'] = max(on_IDX) + 1

        final = datetime.now()
        detection_info.loc[2,'Time_Elapsed'] = (metade - begin) + (final-final2)
        ###############################################
        ##### ----- DETECTION INFO Elliptic ----- #####
        ##### ----------------------------------- #####
        y_pred_elliptic = clf_elliptic.predict(proj_streaming)

        for i,label_idx in enumerate(np.unique(online_labels)):
            if y_pred_elliptic[i] == 1:
                detection_info.loc[3,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[3,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
            else:
                detection_info.loc[3,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index < (b_test)).sum()
                detection_info.loc[3,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == label_idx].index >= (b_test)).sum()
                
        detection_info.loc[3,'N_Groups'] = max(on_IDX) + 1

        final2 = datetime.now()
        detection_info.loc[3,'Time_Elapsed'] = (metade - begin) + (final2-final)


    print("\n\n.Detection Info")
    print(detection_info.to_string(index=False))
    detection_info.to_csv('results/Second_ADP_detection_info_{}_{}.csv'.format(gra,n_i), index=False)

def NEW_ADPOffline_Granularity_Iteration_third(static, streaming, gra, b_test, n_i):
    begin = datetime.now()

    L1, W = static.shape
    L2, _ = streaming.shape

    ##################################
    ##### ----- STATIC ADP ----- #####
    ##### ---------------------- #####

    Input = {'data': static,
             'granularity': gra,
             'distancetype': 'euclidean'}
            
    static_output = ADP.ADP(Input, 'Offline')

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####
            
    IDX = static_output['IDX']
    static_n_centers = max(IDX) + 1

    labeled_static = np.concatenate((static, IDX.reshape(-1,1)), axis=1)

    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    static_df = pd.DataFrame(labeled_static, columns=columns)
    static_cloud_info = []
    for i in range(static_n_centers):
        static_cloud_info.append(HOS_clouds_stats_tau(static_df[static_df['LABEL'] == i]))
    static_clouds = np.array(static_cloud_info)
    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    scaler, pca, proj_static = PCA_clouds(static_clouds, gra)

    #########################################
    ##### ----- TRAIN CLASSIFIERS ----- #####
    ##### ----------------------------- #####
    clf_forest = IsolationForest(contamination=0, bootstrap=True, random_state=0)
    clf_forest.fit(proj_static)

    clf_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_svm.fit(proj_static)

    clf_LOF = LocalOutlierFactor(novelty = True)
    clf_LOF.fit(proj_static)

    clf_elliptic = EllipticEnvelope(contamination=0, random_state=0)
    clf_elliptic.fit(proj_static)
    #####################################
    ##### ----- STREAMING ADP ----- #####
    ##### ------------------------- #####
    # Dreate data frames to save each iteration result.
    detection_info = pd.DataFrame(np.zeros((4,8)), columns=['Method', 'Granularity',
                                                            'True_Positive', 'True_Negative',
                                                            'False_Positive','False_Negative', 
                                                            'N_Groups', 'Time_Elapsed'])

    detection_info.loc[0,'Method'] = 'IsolationForest'
    detection_info.loc[1,'Method'] = 'SVM-rbf'
    detection_info.loc[2,'Method'] = 'LOF'
    detection_info.loc[3,'Method'] = 'EllipticEnvelope'

    detection_info['Granularity'] = gra

    # Execute ADP for Streaming data
    Input = {'data': streaming,
             'granularity': gra,
             'distancetype': 'euclidean'}
    output = ADP.ADP(Input, 'Offline')

    on_center = output['centre']
    on_IDX = output['IDX']

    #########################################################
    ##### ----- DATA CLOUDS STATISTICS ATTRIBUTES ----- #####
    ##### --------------------------------------------- #####

    labeled_streaming = np.concatenate((streaming, on_IDX.reshape(-1,1)), axis=1)
    columns = ['px1', 'py1', 'pz1', 'E1', 'eta1', 'phi1', 'pt1',
           'px2', 'py2', 'pz2', 'E2', 'eta2', 'phi2', 'pt2',
           'Delta_R', 'M12', 'MET', 'S', 'C', 'HT', 'A', 'LABEL']
    streaming_df = pd.DataFrame(labeled_streaming, columns=columns)

    streaming_cloud_info = []
    for i in range(max(on_IDX)+1):
        streaming_cloud_info.append(HOS_clouds_stats_tau(streaming_df[streaming_df['LABEL'] == i]))
    streaming_clouds = np.array(streaming_cloud_info)

    ######################################
    ##### ----- PCA CLOUD INFO ----- #####
    ##### -------------------------- #####
    if True:
        scaled_streaming_clouds = scaler.transform(streaming_clouds)
        proj_streaming = pca.transform(scaled_streaming_clouds)
        metade = datetime.now()
        #############################################
        ##### ----- DETECTION INFO FOREST ----- #####
        ##### --------------------------------- #####
        y_pred_forest = clf_forest.predict(proj_streaming)

        for i in range(len(y_pred_forest)):
            if y_pred_forest[i] == 1:
                detection_info.loc[0,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[0,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
            else:
                detection_info.loc[0,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[0,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()

        detection_info.loc[0,'N_Groups'] = max(on_IDX) + 1

        final = datetime.now()
        detection_info.loc[0,'Time_Elapsed'] = (final - begin)
        ##########################################
        ##### ----- DETECTION INFO SVM ----- #####
        ##### ------------------------------ #####
        y_pred_svm = clf_svm.predict(proj_streaming)

        for i in range(len(y_pred_svm)):
            if y_pred_svm[i] == 1:
                detection_info.loc[1,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[1,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
            else:
                detection_info.loc[1,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[1,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
                
        detection_info.loc[1,'N_Groups'] = max(on_IDX) + 1

        final2 = datetime.now()
        detection_info.loc[1,'Time_Elapsed'] = (metade - begin) + (final2-final)

        ##########################################
        ##### ----- DETECTION INFO LOF ----- #####
        ##### ------------------------------ #####
        y_pred_LOF = clf_LOF.predict(proj_streaming)

        for i in range(len(y_pred_LOF)):
            if y_pred_LOF[i] == 1:
                detection_info.loc[2,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[2,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
            else:
                detection_info.loc[2,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[2,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
                
        detection_info.loc[2,'N_Groups'] = max(on_IDX) + 1

        final = datetime.now()
        detection_info.loc[2,'Time_Elapsed'] = (metade - begin) + (final-final2)
        ###############################################
        ##### ----- DETECTION INFO Elliptic ----- #####
        ##### ----------------------------------- #####
        y_pred_elliptic = clf_elliptic.predict(proj_streaming)

        for i in range(len(y_pred_elliptic)):
            if y_pred_elliptic[i] == 1:
                detection_info.loc[3,'False_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[3,'True_Positive'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
            else:
                detection_info.loc[3,'True_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index < (b_test)).sum()
                detection_info.loc[3,'False_Negative'] += (streaming_df[streaming_df['LABEL'] == i].index >= (b_test)).sum()
                
        detection_info.loc[3,'N_Groups'] = max(on_IDX) + 1

        final2 = datetime.now()
        detection_info.loc[3,'Time_Elapsed'] = (metade - begin) + (final2-final)


    print("\n\n.Detection Info")
    print(detection_info.to_string(index=False))
    detection_info.to_csv('results/Third_ADP_detection_info_{}_{}.csv'.format(gra,n_i), index=False)
