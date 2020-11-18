import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        
        # Checking if we need to peak the same amount of data from each window
        
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

def Normalisation(data):
    """Use standart deviation to normalise the data
    -- Input
    - data
    -- Output
    - Normalised data
    """

    # Normalizing whole data
    scaler = StandardScaler().fit(data)
    norm_data = scaler.transform(data)

    return norm_data

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

    if laplace == True:
        fig.savefig('results/Percentage_of_Variance_Held_laplace.png', bbox_inches='tight') 

    else:
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
 
    if laplace == True:
        fig.savefig('results/Attributes_Contribution_Laplace.png', bbox_inches='tight') 

    else:
        fig.savefig('results/Attributes_Contribution.png', bbox_inches='tight')

    return

def PCA_Projection(background_train,streaming_data, N_PCs, maintained_features=0,laplace=True,):
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

    # Calcules the PCA and projects the data-set into them

    pca= PCA(n_components = N_PCs)
    pca.fit(background_train)
            
    # Calculates the total variance maintained by each PCs
            
    pca_variation = pca.explained_variance_ratio_ * 100
    
    if laplace == True:
        print('Laplace Variation maintained: %.2f' % np.round(pca_variation.sum(), decimals = 2))
    
    else:
        print('Normal Variation maintained: %.2f' % np.round(pca_variation.sum(), decimals = 2))

    proj_background_train = pca.transform(background_train)
    proj_streaming_data = pca.transform(streaming_data)
 

    ### Attributes analyses ###

    if laplace == True:
        columns=list(maintained_features)

    else:
        columns=["px1","py1","pz1","E1","eta1","phi1","pt1",\
                    "px2","py2","pz2","E2","eta2","phi2",\
                    "pt2","Delta_R","M12","MET","S","C","HT",\
                    "A", "Min","Max","Mean","Var","Skw","Kurt",\
                    "M2","M3","M4","Bmin","Bmax"]

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

def statistics_attributes(data,xyz_attributes=True):
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
       
    When xyz_attributes=False:
       Concatenate with the data, statistics attributes for each event.
       Only for momentum components (columns [0,1,2,7,8,9])
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
       - output_data = data with adition of statistical features for momentum each line [numpy.array]"""
    if xyz_attributes == True:
        momentum_data = np.concatenate((data[:, 0:3], data[:, 7:10]), axis=1)

        L, W = data.shape

        _, (min_v, max_v), mean, var, skew, kurt = stats.describe(momentum_data.transpose())
        
        # Minimum Value
        min_v = min_v.reshape(-1,1)
        
        # Maximum Value
        max_v = max_v.reshape(-1,1)
        
        # Mean
        mean = mean.reshape(-1,1)
        
        # Variance
        var = var.reshape(-1,1)
        
        # Skewness
        skew = skew.reshape(-1,1)
        
        # Kurtosis
        kurt = kurt.reshape(-1,1)
        
        # 2nd Central Moment
        moment2 = stats.moment(momentum_data.transpose(), moment=2).reshape(-1,1)
        
        # 3rd Central Moment
        moment3 = stats.moment(momentum_data.transpose(), moment=3).reshape(-1,1)
        
        # 4th Central Moment
        moment4 = stats.moment(momentum_data.transpose(), moment=4).reshape(-1,1)
        
        bayes_min = np.zeros(L).reshape(-1,1)
        bayes_max = np.zeros(L).reshape(-1,1)
        for i,d in enumerate(momentum_data):
            bayes = stats.bayes_mvs(d)
            bayes_min[i] = bayes[0][1][0]
            bayes_max[i] = bayes[0][1][1]
    
    else: 
        L, W = data.shape

        _, (min_v, max_v), mean, var, skew, kurt = stats.describe(data.transpose())
        
        # Minimum Value
        min_v = min_v.reshape(-1,1)
        
        # Maximum Value
        max_v = max_v.reshape(-1,1)
        
        # Mean
        mean = mean.reshape(-1,1)
        
        # Variance
        var = var.reshape(-1,1)
        
        # Skewness
        skew = skew.reshape(-1,1)
        
        # Kurtosis
        kurt = kurt.reshape(-1,1)
        
        # 2nd Central Moment
        moment2 = stats.moment(data.transpose(), moment=2).reshape(-1,1)
        
        # 3rd Central Moment
        moment3 = stats.moment(data.transpose(), moment=3).reshape(-1,1)
        
        # 4th Central Moment
        moment4 = stats.moment(data.transpose(), moment=4).reshape(-1,1)
        
        # Bayesian Confidence Interval
        bayes_min = np.zeros(L).reshape(-1,1)
        bayes_max = np.zeros(L).reshape(-1,1)
        for i,d in enumerate(data):
            bayes = stats.bayes_mvs(d)
            bayes_min[i] = bayes[0][1][0]
            bayes_max[i] = bayes[0][1][1]
    
    
    output_data = np.concatenate((data, min_v, max_v, mean,
                                        var, skew, kurt,
                                        moment2, moment3, moment4,
                                        bayes_min, bayes_max), axis=1)
    return output_data


def SODA_Granularity_Iteration(offline_data,streaming_data,gra,n_backgound,Iteration,laplace):
    ## Formmating  Data
    offline_data = np.matrix(offline_data)
    L1 = len(offline_data)

    streaming_data = np.matrix(streaming_data)
    
    data = np.concatenate((offline_data, streaming_data), axis=0)

    # Dreate data frames to save each iteration result.

    detection_info = pd.DataFrame(np.zeros((1,6)).reshape((1,-1)), columns=['Granularity','True_Positive', 'True_Negative','False_Positive','False_Negative', 'N_Groups'])

    performance_info = pd.DataFrame(np.zeros((1,8)).reshape((1,-1)), columns=['Granularity', 'Time_Elapsed',
                                                                'Mean CPU_Percentage', 'Max CPU_Percentage',
                                                                'Mean RAM_Percentage', 'Max RAM_Percentage',
                                                                'Mean RAM_Usage_GB', 'Max RAM_Usage_GB'])

    begin = datetime.now()

    performance_thread = performance()
    performance_thread.start()
    
    detection_info.loc[0,'Granularity'] = gra
    performance_info.loc[0,'Granularity'] = gra

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

    performance_thread.stop()
    performance_out = performance_thread.join()
    final = datetime.now()
    performance_info.loc[0,'Time_Elapsed'] = (final - begin)
    performance_info.loc[0,'Mean CPU_Percentage'] = performance_out['mean_cpu_p']
    performance_info.loc[0,'Max CPU_Percentage'] = performance_out['max_cpu_p']
    performance_info.loc[0,'Mean RAM_Percentage'] = performance_out['mean_ram_p']
    performance_info.loc[0,'Max RAM_Percentage'] = performance_out['max_ram_p']
    performance_info.loc[0,'Mean RAM_Usage_GB'] = performance_out['mean_ram_u']
    performance_info.loc[0,'Max RAM_Usage_GB'] = performance_out['max_ram_u']

    if laplace == 0:
        detection_info.to_csv('results/detection_info_Laplace' + str(gra) + '_' + str(Iteration) + '.csv', index=False)
        performance_info.to_csv('results/performance_info_Laplace' + str(gra) + '_' + str(Iteration) + '.csv', index=False)
    
    else:
        detection_info.to_csv('results/detection_info_raw_' + str(gra) + '_' + str(Iteration) + '.csv', index=False)
        performance_info.to_csv('results/performance_info_raw_' + str(gra) + '_' + str(Iteration) + '.csv', index=False)

def construct_W(X, neighbour_size = 5, t = 1):
    n_samples, n_features = np.shape(X)
    S=kneighbors_graph(X, neighbour_size+1, mode='distance',metric='euclidean')
    S = (-1*(S*S))/(2*t*t)
    S=S.tocsc()
    S=expm(S) # exponential
    S=S.tocsr()
    #[1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
    #Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method
    bigger = np.transpose(S) > S
    S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
    return S

def LaplacianScore(X, neighbour_size = 5,  t = 1):
    W = construct_W(X,t=t,neighbour_size=neighbour_size)
    n_samples, n_features = np.shape(X)
    
    #construct the diagonal matrix
    D=np.array(W.sum(axis=1))
    D = scipy.sparse.diags(np.transpose(D), [0])
    #construct graph Laplacian L
    L=D-W.toarray()

    #construct 1= [1,···,1]' 
    I=np.ones((n_samples,n_features))

    #construct fr' => fr= [fr1,...,frn]'
    Xt = np.transpose(X)

    #construct fr^=fr-(frt D I/It D I)I
    t=np.matmul(np.matmul(Xt,D.toarray()),I)/np.matmul(np.matmul(np.transpose(I),D.toarray()),I)
    t=t[:,0]
    t=np.tile(t,(n_samples,1))
    fr=X-t

    #Compute Laplacian Score
    fr_t=np.transpose(fr)
    Lr=np.matmul(np.matmul(fr_t,L),fr)/np.matmul(np.dot(fr_t,D.toarray()),fr)

    return np.diag(Lr)

def distanceEntropy(d, mu = 0.5, beta=10):
    """
    As per: An Unsupervised Feature Selection Algorithm: Laplacian Score Combined with
    Distance-based Entropy Measure, Rongye Liu 
    """
    if d<=mu:
        result = (np.exp(beta * d) - np.exp(0))/(np.exp(beta * mu) - np.exp(0))
    else:
        result = (np.exp(beta * (1-d) )- np.exp(0))/(np.exp(beta *(1- mu)) - np.exp(0))              
    return result

def lse(data, ls):
    """
    This method takes as input a dataset, its laplacian scores for all features
    and applies distance based entropy feature selection in order to identify
    the best subset of features in the laplacian sense.
    """
    orderedFeatures = np.argsort(ls)
    scores = {}
    for i in range (2,len(ls)):
        selectedFeatures = orderedFeatures[:i]
        selectedFeaturesDataset = data[:, selectedFeatures]
        d =sklearn.metrics.pairwise_distances(selectedFeaturesDataset, metric = 'euclidean' )
        beta =10
        mu = 0.5

        d = preprocessing.MinMaxScaler().fit_transform(d)
        e = np.vectorize(distanceEntropy)(d) 
        e = preprocessing.MinMaxScaler().fit_transform(e)
        totalEntropy= np.sum(e)
        scores[i] = totalEntropy
    bestFeatures = orderedFeatures[:list(scores.keys())[np.argmin(scores.values())]]
    return bestFeatures

def laplacian_score(xyz_background, xyz_signal, n_dimensions):

    # Calculate Laplace Score for the background and signal
    laplace_background = LaplacianScore(xyz_background)
    laplace_signal = LaplacianScore(xyz_signal)

    laplace_background = laplace_background.reshape((1,-1))
    laplace_signal = laplace_signal.reshape((1,-1))

    # Creating Data frames for the laplace score
    laplace_background_df = pd.DataFrame (laplace_background, columns=["px1","py1","pz1","E1","eta1","phi1","pt1",\
                                        "px2","py2","pz2","E2","eta2","phi2",\
                                        "pt2","Delta_R","M12","MET","S","C","HT",\
                                        "A", "Min","Max","Mean","Var","Skw","Kurt",\
                                        "M2","M3","M4","Bmin","Bmax"])
    laplace_signal_df = pd.DataFrame (laplace_signal, columns=laplace_background_df.columns)

    # Sorting backgorund attributes by their importance values 
    laplace_background_df = laplace_background_df.sort_values(by=0, axis=1,ascending=False)
    
    # Sorting signal attributes regarding the background laplace score 
    laplace_signal_df = laplace_signal_df[laplace_background_df.columns]

    # Maintening the defined number of determined features
    maintained_features = list(laplace_background_df.columns[:n_dimensions])

    
    main_laplace_background = pd.DataFrame(np.zeros((n_dimensions)).reshape((1,-1)),columns=maintained_features)
    main_laplace_signal_df = pd.DataFrame(np.zeros((n_dimensions)).reshape((1,-1)),columns=maintained_features)

    for col in maintained_features:
        main_laplace_background.loc[0,col] = laplace_background_df.loc[0,col]
        main_laplace_signal_df.loc[0,col] = laplace_signal_df.loc[0,col]
    
    # Ploting Results
    
    sorted_laplace_background = main_laplace_background.values[:]      
    sorted_laplace_signal = main_laplace_signal_df.values[:]      

    # Ploting backgound's Attributes importance

    fig = plt.figure(figsize=[20,10])

    fig.suptitle('Laplacian backgound\'s features importance scores', fontsize=20)

    ax = fig.subplots(1,1)

    sorted_laplace_background = sorted_laplace_background.ravel()
    ax.bar(x=list(main_laplace_background.columns),height=sorted_laplace_background)
    plt.ylabel('Relevance',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()    
    fig.savefig('results/Laplacian_Background_Score.png', bbox_inches='tight') 
    
    # Ploting signal's Attributes importance

    fig = plt.figure(figsize=[20,10])

    fig.suptitle('Laplacian signal\'s features importance scores', fontsize=20)

    ax = fig.subplots(1,1)

    sorted_laplace_signal = sorted_laplace_signal.ravel()
    ax.bar(x=list(main_laplace_signal_df.columns),height=sorted_laplace_signal)
    plt.ylabel('Relevance',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    fig.savefig('results/Laplacian_Signal_Score.png', bbox_inches='tight') 

    return maintained_features

def laplacian_score_2(xyz_background, xyz_signal):

    # Calculate Laplace Score for the background and signal
    
    laplace_background = LaplacianScore(xyz_background)
    laplace_signal = LaplacianScore(xyz_signal)
    
    # Apply Entropy to select features
    
    selectedFeatures = lse(xyz_background, laplace_background)

    print('Number of Selected Features: ' + str(len(selectedFeatures)))
    
    # Reshape and sort matrixs to create data-frames
    laplace_background = laplace_background.reshape((1,-1))
    laplace_signal = laplace_signal.reshape((1,-1))

    # Creating Data frames for the laplace score
    laplace_background_df = pd.DataFrame (laplace_background, columns=["px1","py1","pz1","E1","eta1","phi1","pt1",\
                                        "px2","py2","pz2","E2","eta2","phi2",\
                                        "pt2","Delta_R","M12","MET","S","C","HT",\
                                        "A", "Min","Max","Mean","Var","Skw","Kurt",\
                                        "M2","M3","M4","Bmin","Bmax"])
    laplace_signal_df = pd.DataFrame (laplace_signal, columns=laplace_background_df.columns)

    # Sorting backgorund attributes by their importance values 
    laplace_background_df = laplace_background_df.sort_values(by=0, axis=1,ascending=True)
    
    # Sorting signal attributes regarding the background laplace score 
    laplace_signal_df = laplace_signal_df[laplace_background_df.columns]     
    
    # Maintening the defined number of determined features
    maintained_features = list(laplace_background_df.columns[:len(selectedFeatures)])
    
    main_laplace_background = pd.DataFrame(np.zeros((len(selectedFeatures))).reshape((1,-1)),columns=maintained_features)
    main_laplace_signal_df = pd.DataFrame(np.zeros((len(selectedFeatures))).reshape((1,-1)),columns=maintained_features)

    for col in maintained_features:
        main_laplace_background.loc[0,col] = laplace_background_df.loc[0,col]
        main_laplace_signal_df.loc[0,col] = laplace_signal_df.loc[0,col]
    
    # Ploting Results
    
    sorted_laplace_background = main_laplace_background.values[:]      
    sorted_laplace_signal = main_laplace_signal_df.values[:]      

    # Ploting backgound's Attributes importance

    fig = plt.figure(figsize=[20,10])

    fig.suptitle('Laplacian backgound\'s features importance scores', fontsize=20)

    ax = fig.subplots(1,1)

    sorted_laplace_background = sorted_laplace_background.ravel()
    ax.bar(x=list(main_laplace_background.columns),height=sorted_laplace_background)
    plt.ylabel('Relevance',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()    
    fig.savefig('results/Laplacian_Background_Score.png', bbox_inches='tight') 
    
    # Ploting signal's Attributes importance

    fig = plt.figure(figsize=[20,10])

    fig.suptitle('Laplacian signal\'s features importance scores', fontsize=20)

    ax = fig.subplots(1,1)

    sorted_laplace_signal = sorted_laplace_signal.ravel()
    ax.bar(x=list(main_laplace_signal_df.columns),height=sorted_laplace_signal)
    plt.ylabel('Relevance',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    fig.savefig('results/Laplacian_Signal_Score.png', bbox_inches='tight') 

    return maintained_features
def laplacian_reduction(data,maintained_features):
    
    # Creating data frame for the original data attibutes
    data_df = pd.DataFrame (data, columns=["px1","py1","pz1","E1","eta1","phi1","pt1",\
                                        "px2","py2","pz2","E2","eta2","phi2",\
                                        "pt2","Delta_R","M12","MET","S","C","HT",\
                                        "A", "Min","Max","Mean","Var","Skw","Kurt",\
                                        "M2","M3","M4","Bmin","Bmax"])
    
    # Creating data frame with the selected features 
    maintained_data = pd.DataFrame (np.zeros((len(data),len(maintained_features))),columns=maintained_features)
    
    for col in maintained_features:
        maintained_data.loc[:,col] = data_df.loc[:,col]
        
    return maintained_data