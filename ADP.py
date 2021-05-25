import numpy as np
import pandas as pd
import scipy.io
import scipy.spatial.distance
import numpy.matlib
from datetime import datetime
from scipy.spatial.distance import pdist, cdist, squareform

def OfflineADP(Input): 
    #distancetype changeable, i.e. 'euclidean','cityblock','sqeuclidean','cosine'.
    data = Input['data']
    granularity = Input['granularity']
    distancetype = Input['distancetype']

    L0,W0=data.shape
    udata,frequency = np.unique(data,return_counts=True, axis=0)
    frequency=np.matrix(frequency)
    L,W=udata.shape
    dist=(scipy.spatial.distance.cdist(udata,udata,metric=distancetype))**2
    unidata_pi=np.sum(np.multiply(dist,np.matlib.repmat(frequency,L,1)), axis=1)
    unidata_density=np.transpose(unidata_pi)*np.transpose(frequency)/(unidata_pi*2*L0)
    unidata_Gdensity=np.multiply(unidata_density,np.transpose(frequency))
    samples_Gdensity = unidata_Gdensity.copy()

    pos = np.zeros(L)
    pos[0] = int(np.argmax(unidata_Gdensity))
    seq = np.array(range(0,L))
    seq = np.delete(seq,int(pos[0]))

    for ii in range(1,L):
        p1=np.argmin(dist[int(pos[ii-1]),seq])
        pos[ii]=seq[p1]
        seq=np.delete(seq,p1)

    udata2=np.zeros([L,W])
    uGD=np.zeros(L)
    for ii in range(0,L):
        udata2[ii,:]=udata[int(pos[ii]),:]
        uGD[ii]=unidata_Gdensity[int(pos[ii]),0]

    uGD1=uGD[range(0,L-2)]-uGD[range(1,L-1)]
    uGD2=uGD[range(1,L-1)]-uGD[range(2,L)]
    seq2=np.array(range(1,L-1))

    seq3=np.array([i for i in range(len(uGD2)) if uGD1[i]<0 and uGD2[i]>0])

    seq4=np.array([0])
    if uGD2[L-3]<0:
        seq4=np.append(seq4,seq2[seq3])
        seq4=np.append(seq4,np.array([int(L-1)]))
    else:
        seq4=np.append(seq4,seq2[seq3])

    L2, =seq4.shape
    centre0=np.zeros([L2,W])
    for ii in range(0,L2):
        centre0[ii]=udata2[int(seq4[ii]),:]
    
    dist1=scipy.spatial.distance.cdist(data,centre0,metric=distancetype)
    seq5=dist1.argmin(1)
    centre1=np.zeros([L2,W])
    Mnum=np.zeros(L2)

    for ii in range(0,L2):
        seq6=[i for i in range(len(seq5)) if seq5[i] == ii]
        Mnum[ii]=len(seq6)
        centre1[ii,:]=np.mean(data[seq6,:],axis=0)

    seq7=[i for i in range(len(Mnum)) if Mnum[i] > 1]
    seq8=[i for i in range(len(Mnum)) if Mnum[i] <= 1]

    L3=len(seq7)
    L4=len(seq8)

    centre2=np.zeros([L3,W])
    centre3=np.zeros([L4,W])
    Mnum1=np.zeros(L3)
    for ii in range(0,L3):
        centre2[ii,:]=centre1[seq7[ii],:]
        Mnum1[ii]=Mnum[seq7[ii]]

    for ii in range(0,L4):
        centre3[ii,:]=centre1[seq8[ii],:]

    dist2=scipy.spatial.distance.cdist(centre3,centre2,distancetype)
    seq9=dist2.argmin(1)

    for ii in range(0,L4):
        centre2[seq9[ii],:]=centre2[seq9[ii],:]*Mnum1[seq9[ii]]/(Mnum1[seq9[ii]]+1)+centre3[ii,:]/(Mnum1[seq9[ii]]+1)
        Mnum1[seq9[ii]]=Mnum1[seq9[ii]]+1

    UD2=centre2
    L5=0
    Count=0
    while L5 != L3 and L3>2:
        Count=Count+1
        L5=L3
        dist3=scipy.spatial.distance.cdist(data,UD2,distancetype)
        seq10=dist3.argmin(1)
        centre3=np.zeros([L3,W])
        Mnum3=np.zeros(L3)
        Sigma3=np.zeros(L3)
        seq12=[]
        for ii in range(0,L3):
            seq11=[i for i in range(len(seq10)) if seq10[i] == ii]
            if len(seq11)>=2:
                data1=data[seq11,:]
                Mnum3[ii]=len(seq11)
                centre3[ii,:]=np.sum(data1,axis=0)/Mnum3[ii]
                Sigma3[ii]=np.sum(np.sum(np.multiply(data1,data1)))/Mnum3[ii]-np.sum(np.multiply(centre3[ii,:],centre3[ii,:]))
                if Sigma3[ii]>0:
                    seq12.append(ii)
        L3=len(seq12)
        Mnum3=np.matrix(Mnum3[seq12])
        centre3=centre3[seq12,:]
        dist=(scipy.spatial.distance.cdist(centre3,centre3,distancetype))**2
        unidata_pi=np.sum(np.multiply(dist,np.matlib.repmat(Mnum3,L3,1)), axis=1)
        unidata_density=np.transpose(unidata_pi)*np.transpose(Mnum3)/(unidata_pi*2*L0)
        unidata_Gdensity=np.multiply(unidata_density,np.transpose(Mnum3))
        dist2=(scipy.spatial.distance.pdist(centre3,distancetype))
        dist3=scipy.spatial.distance.squareform(dist2)
        Aver1=np.mean(dist2)
        for ii in range(granularity):
            Aver1=np.mean(dist2[dist2<=Aver1])
        Sigma=Aver1/2
        dist3=dist3-np.ones([L3,L3])*Sigma
        seq15=[]
        for i in range(0,L3):
            seq13=np.array(list(range(0,i))+list(range(i+1,L3)))
            seq14=seq13[dist3[i,seq13]<0]
            if len(seq14)>0:
                if unidata_Gdensity[i]>max(unidata_Gdensity[seq14]):
                    seq15.append(i)
            else:
                seq15.append(i)
        L3=len(seq15)
        UD2=centre3[np.array(seq15),:]

    centre=UD2

    dist1=scipy.spatial.distance.cdist(data,centre,distancetype)
    IDX=dist1.argmin(1)
    Mnum=np.zeros(L3)

    for ii in range(0,L3):
        seq6=[i for i in range(len(IDX)) if IDX[i] == ii]
        Mnum[ii]=len(seq6)
        centre[ii,:]=np.sum(data[seq6,:],axis=0)/Mnum[ii]

    Global_X = np.mean(np.sum(data**2, axis=1), axis=0)
    Param = {'ModelNumber':L3,
             'Support':Mnum, 
             'Center':centre,
             'Global_X':Global_X,
             'Global_mean':np.mean(data,axis=0),
             'K':L0}

    output = {'centre': centre,
              'IDX': IDX,
              'Param':Param,
              'CloudsGlobalDensity':unidata_Gdensity,
              'SamplesGlobalDensity':samples_Gdensity}
    return output

def HybridVersion(data,Global_mean,Global_X,ModelNumber,center,Support,K):
    L, W = data.shape
    L = L + K
    
    for ii in range(K, L):
        Global_mean = ii/(ii+1)*Global_mean+data[ii-K,:]/(ii+1)
        Global_X = ii/(ii+1)*Global_X+np.sum(data[ii-K,:]**2)/(ii+1)
        dist = cdist(np.vstack((data[ii-K,:],center)), Global_mean.reshape(1,-1), 'euclidean')
        VC = np.sqrt(2*(Global_X-np.sum(Global_mean**2)))
        SIGMA = VC/2
        if np.min(dist[1:ModelNumber+1]) > dist[0] or np.max(dist[1:ModelNumber+1]) < dist[0]:
            ModelNumber += 1
            Support = np.insert(Support,len(Support),1)
            center = np.vstack((center, data[ii-K,:]))
        else:
            dist2 = cdist(data[ii-K,:].reshape(1,-1), center, 'euclidean')
            V = np.min(dist2)
            pos = np.argmin(dist2)
            if V<SIGMA:
                center[pos,:] = center[pos,:]*Support[pos]/(Support[pos]+1) + data[ii-K,:]/(Support[pos]+1)
                Support[pos] += 1
            else:
                ModelNumber += 1
                Support = np.insert(Support,len(Support),1)
                center = np.vstack((center, data[ii-K,:]))

    return ModelNumber,center,Support,Global_mean,Global_X,L

def RemoveAbnormalDataClouds(ModelNumber,center,Support):
    seq0 = np.argwhere(Support==1)
    M0 = len(seq0)
    ModelNumber = ModelNumber-M0
    C0 = center[seq0,:]
    seq0 = np.argwhere(Support>1)
    center = center[seq0,:]
    Support = Support[seq0]
    x1,x2,x3 = C0.shape
    C0 = C0.reshape(x1,x3)
    x1,x2,x3 = center.shape
    center = center.reshape(x1,x3)
    dist0 = cdist(C0, center, 'euclidean')
    ident = np.argmin(dist0, axis=1)
    for i in range(M0):
        Support[ident[i]] += 1
        center[ident[i],:] = center[ident[i],:]*Support[ident[i]]/(Support[ident[i]]+1) + C0[i,:]/(Support[ident[i]]+1)
    
    return ModelNumber,center

def FormingDataCloud(centre,data):
    _, W = centre.shape
    dist=cdist(data,centre,'euclidean')
    L, C = dist.shape
    
    IDX = np.argmin(dist, axis=1)
    Support = np.zeros((C,1))
    center = np.zeros((C,W))
    for i in range(C):
        seq = np.argwhere(IDX==i)
        Support[i] = len(seq)
        center[i,:] = np.mean(data[seq,:], axis=0)
    
    return C,center,IDX,Support

def HybridADP(Input):
    data = Input['newData']
    data2 = Input['historicalData']
    Param = Input['SystemParams']
    ModelNumber = Param['ModelNumber']
    center = Param['Center']
    Support = Param['Support']
    Global_X = Param['Global_X']
    Global_mean = Param['Global_mean']
    K = Param['K']
    
    ModelNumber,center,Support,Global_mean,Global_X,L=HybridVersion(data,Global_mean,Global_X,ModelNumber,center,Support,K)
    
    Param = {'ModelNumber': ModelNumber,
             'Center': center,
             'Support': Support,
             'Global_X': Global_X,
             'Global_mean': Global_mean,
             'K': L}
    
    _,center = RemoveAbnormalDataClouds(ModelNumber,center,Support)
    _,centre,IDX,_ = FormingDataCloud(center,np.vstack((data,data2)))
    
    output = {'centre': centre,
              'IDX': IDX,
              'Param':Param}
    
    return output

def EvolvingVersion(data):
    L, W = data.shape
    Global_mean = data[0,:]
    Global_X = np.sum(data[0,:]**2)
    center = data[0,:]
    Support = [1]
    ModelNumber = 1
    
    for ii in range(1,L):
        Global_mean = ii/(ii+1)*Global_mean+data[ii,:]/(ii+1)
        Global_X = ii/(ii+1)*Global_X+np.sum(data[ii,:]**2)/(ii+1)
        dist = cdist(np.vstack((data[ii,:],center)), Global_mean.reshape(1,-1), 'euclidean')
        VC = np.sqrt(2*(Global_X-np.sum(Global_mean**2)))
        SIGMA = VC/2        
        if np.min(dist[1:ModelNumber+1]) > dist[0] or np.max(dist[1:ModelNumber+1]) < dist[0]:
            ModelNumber += 1
            Support = np.insert(Support,len(Support),1)
            center = np.vstack((center, data[ii,:]))
        else:
            dist2 = cdist(data[ii,:].reshape(1,-1), center, 'euclidean')
            V = np.min(dist2)
            pos = np.argmin(dist2)
            if V<SIGMA:
                center[pos,:] = center[pos,:]*Support[pos]/(Support[pos]+1) + data[ii,:]/(Support[pos]+1)
                Support[pos] += 1
            else:
                ModelNumber += 1
                Support = np.insert(Support,len(Support),1)
                center = np.vstack((center, data[ii,:]))
    
    return ModelNumber,center,Support,Global_mean,Global_X,L

def EvolvingADP(Input):
    data = Input['StreamingData']
    
    ModelNumber,center,Support,Global_mean,Global_X,L = EvolvingVersion(data)
    
    Param = {'ModelNumber': ModelNumber,
             'Center': center,
             'Support': Support,
             'Global_X': Global_X,
             'Global_mean': Global_mean,
             'K': L}
    
    _,center = RemoveAbnormalDataClouds(ModelNumber,center,Support)
    _,centre,IDX,_ = FormingDataCloud(center,data)
    
    output = {'centre': centre,
              'IDX': IDX,
              'Param':Param}
    return output

def ADP(Input, mode):
	if mode == 'Offline':
		output = OfflineADP(Input)
	elif mode == 'Hybrid':
		output = HybridADP(Input)
	elif mode == 'Evolving':
		output = EvolvingADP(Input)
	else:
		print(mode, "does not exist!\n Available options ['Offline', 'Hybrid', 'Evolving']")

	return output
