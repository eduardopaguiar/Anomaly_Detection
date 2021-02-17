from scipy.spatial.distance import pdist, cdist, squareform
import math as mt
import pandas as pd
import numpy as np
from datetime import datetime
from numba import njit,jit
from numba.typed import List
import numba as nb
import multiprocessing as mp
import pickle

def grid_set(data, N):
    _ , W = data.shape
    AvD1 = data.mean(0)
    X1 = np.mean(np.sum(np.power(data,2),axis=1))
    grid_trad = np.sqrt(2*(X1 - AvD1*AvD1.T))/N
    Xnorm = np.sqrt(np.sum(np.power(data,2),axis=1))
    aux = Xnorm
    for i in range(W-1):
        aux = np.insert(aux,0,Xnorm.T,axis=1)
    data = data / aux
    seq = np.argwhere(np.isnan(data))
    if tuple(seq[::]): data[tuple(seq[::])] = 1
    AvD2 = data.mean(0)
    grid_angl = np.sqrt(1-AvD2*AvD2.T)/N
    return X1, AvD1, AvD2, grid_trad, grid_angl

def pi_calculator(Uniquesample, mode):
    UN, W = Uniquesample.shape
    if mode == 'euclidean' or mode == 'mahalanobis' or mode == 'cityblock' or mode == 'chebyshev' or mode == 'canberra':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = []
        for i in range(UN): aux.append(AA1)
        aux2 = [Uniquesample[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.power(aux2,2),axis=1)+DT1

        
    if mode == 'minkowski':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = np.matrix(AA1)
        for i in range(UN-1): aux = np.insert(aux,0,AA1,axis=0)
        aux = np.array(aux)
        
        uspi = np.power(cdist(Uniquesample, aux, mode, p=1.5),2)+DT1
        uspi = uspi[:,0]

    if mode == 'cosine':
        Xnorm = np.matrix(np.sqrt(np.sum(np.power(Uniquesample,2),axis=1))).T
        aux2 = Xnorm
        for i in range(W-1):
            aux2 = np.insert(aux2,0,Xnorm.T,axis=1)
        Uniquesample1 = Uniquesample / aux2
        AA2 = np.mean(Uniquesample1,0)
        X2 = 1
        DT2 = X2 - np.sum(np.power(AA2,2))
        aux = []
        for i in range(UN): aux.append(AA2)
        aux2 = [Uniquesample1[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.sum(np.power(aux2,2),axis=1),axis=1)+DT2
        
    return uspi

def Globaldensity_Calculator(data, distancetype):
    Uniquesample, J, K = np.unique(data, axis=0, return_index=True, return_inverse=True)
    Frequency, _ = np.histogram(K,bins=len(J))
    uspi1 = pi_calculator(Uniquesample, distancetype)

    sum_uspi1 = sum(uspi1)
    Density_1 = uspi1 / sum_uspi1

    uspi2 = pi_calculator(Uniquesample, 'cosine')

    sum_uspi2 = sum(uspi2)
    Density_2 = uspi1 / sum_uspi2
    GD = (Density_2+Density_1) * Frequency
    index = GD.argsort()[::-1]
    GD = GD[index]
    Uniquesample = Uniquesample[index]
    Frequency = Frequency[index]

    return GD, Uniquesample, Frequency

@njit
def chessboard_division_njit(Uniquesample, MMtypicality, interval1, interval2, distancetype):
    L, W = Uniquesample.shape
    if distancetype == 'euclidean':
        W = 1

    BOX = [Uniquesample[k] for k in range(W)]
    BOX_miu = [Uniquesample[k] for k in range(W)]
    BOX_S = [1]*W
    BOX_X = [np.sum(Uniquesample[k]**2) for k in range(W)]
    NB = W
    BOXMT = List([MMtypicality[k] for k in range(W)])

    for i in range(W,L):
        XA = Uniquesample[i].reshape(1,-1)
        XB = BOX_miu
        a = [] # Euclidean
        b = [] # Cosine
        for ii in range (len(XA)):
            aux2 = [] # Euclidean
            aux3 = [] # Cosine
            for j in range (len(XB)):
                aux1 = [] # Euclidean
                bux1 = 0 # Euclidean
                dot = 0 # Cosine
                denom_a = 0 # Cosine
                denom_b = 0 # Cosine
                for k in range (len(XB[j])):
                    aux1.append((XB[j][k]-XA[ii,k])**2) # Euclidean
                    bux1 += ((XB[j][k]-XA[ii,k])**2) # Euclidean
                    dot += (XB[j][k]*XA[ii,k]) # Cosine
                    denom_a += (XB[j][k] * XB[j][k]) # Cosine
                    denom_b += (XA[ii,k] * XA[ii,k]) # Cosine
                aux2.append(bux1**(0.5)) # Euclidean
                d2 = (1 - ((dot / ((denom_a ** 0.5) * (denom_b ** 0.5))))) # Cosine
                if d2 < 0:
                    aux3.append(0) # Cosine
                else:
                    aux3.append(d2**0.5) # Cosine
            b.append(aux3) # Cosine
            a.append(aux2) # Euclidean
        distance = np.array([a[0],b[0]]).T
        
        SQ = []
        for j,d in enumerate(distance):
            if d[0] < interval1 and d[1] < interval2:
                SQ.append(j)
        COUNT = len(SQ)

        if COUNT == 0:
            BOX.append(Uniquesample[i])
            NB = NB + 1
            BOX_S.append(1)
            BOX_miu.append(Uniquesample[i])
            BOX_X.append(np.sum(Uniquesample[i]**2))
            BOXMT.append(MMtypicality[i])

        if COUNT >= 1:
            DIS = [distance[S,0]/interval1[0] + distance[S,1]/interval2[0] for S in SQ]# pylint: disable=E1136  # pylint/issues/3139
            b = 0
            mini = DIS[0]
            for ii in range(1,len(DIS)):
                if DIS[ii] < mini:
                    mini = DIS[ii]
                    b = ii
            BOX_S[SQ[b]] = BOX_S[SQ[b]] + 1
            BOX_miu[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_miu[SQ[b]] + Uniquesample[i]/BOX_S[SQ[b]]
            BOX_X[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_X[SQ[b]] + np.sum(Uniquesample[i]**2)/BOX_S[SQ[b]]
            BOXMT[SQ[b]] = BOXMT[SQ[b]] + MMtypicality[i] 

    return BOX, BOX_miu, BOX_X, BOX_S, BOXMT, NB

def ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,Internval1,Internval2, distancetype):
    Centers = []
    n = 2
    ModeNumber = 0
    
    if distancetype == 'minkowski':
        distance1 = squareform(pdist(BOX_miu,metric=distancetype, p=1.5))
    else:
        distance1 = squareform(pdist(BOX_miu,metric=distancetype))  
    
    distance2 = np.sqrt(squareform(pdist(BOX_miu,metric='cosine')))

    for i in range(NB):
        seq = []
        for j,(d1,d2) in enumerate(zip(distance1[i],distance2[i])):
            if d1 < n*Internval1 and d2 < n*Internval2:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]
        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1
    return Centers, ModeNumber

@njit
def ChessBoard_PeakIdentification_njit(BOX_miu,BOXMT,NB,Internval1,Internval2, distancetype):
    Centers = []
    n = 2
    ModeNumber = 0
    L, W = BOX_miu.shape
    
    for i in range(L):
        distance1 = np.zeros((L)) # Euclidean
        distance2 = np.zeros((L)) # Cosine
        for j in range(L):
            aux = 0 # Euclidean
            num = 0 # Cosine
            den1 = 0 # Cosine
            den2 = 0 # Cosine
            for k in range(W):
                aux += (BOX_miu[i,k] - BOX_miu[j,k])**2 # Euclidean
                num += BOX_miu[i,k]*BOX_miu[j,k] # Cosine
                den1 += BOX_miu[i,k]**2 # Cosine
                den2 += BOX_miu[j,k]**2 # Cosine
            distance1[j] = aux**.5 # Euclidean
            dis2 = (1 - num/(den1**.5 * den2**.5) ) # Cosine
            if dis2 < 0:
                distance2[j] = 0 # Cosine
            else:
                distance2[j] = dis2**.5 # Cosine

        seq = []
        for j,(d1,d2) in enumerate(zip(distance1,distance2)):
            if d1 < n*Internval1 and d2 < n*Internval2:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]
        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1
    return Centers, ModeNumber

@njit
def cloud_member_recruitment_njit(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    L, W = Uniquesample.shape
    Membership = np.zeros((L,ModelNumber))
    Members = np.zeros((L,ModelNumber*W))
    Count = []
    
    distance1 = [] # Euclidean
    distance2 = [] # Cosine
    for ii in range(L):
        aux2 = [] # Euclidean
        aux3 = [] # Cosine
        for j in range (len(Center_samples)):
            aux1 = [] # Euclidean
            bux1 = 0 # Euclidean
            dot = 0 # Cosine
            denom_a = 0 # Cosine
            denom_b = 0 # Cosine
            for k in range(W):
                aux1.append((Center_samples[j][k]-Uniquesample[ii,k])**2) # Euclidean
                bux1 += ((Center_samples[j][k]-Uniquesample[ii,k])**2) # Euclidean
                dot += (Center_samples[j][k]*Uniquesample[ii,k]) # Cosine
                denom_a += (Center_samples[j][k] * Center_samples[j][k]) # Cosine
                denom_b += (Uniquesample[ii,k] * Uniquesample[ii,k]) # Cosine

            aux2.append((bux1**(0.5))/grid_trad) # Euclidean
            d2 = (1 - ((dot / ((denom_a ** 0.5) * (denom_b ** 0.5))))) # Cosine
            if d2 < 0:
                aux3.append(0) # Cosine
            else:
                aux3.append((d2**0.5)/grid_angl) # Cosine

        distance1.append(aux2) # Euclidean
        distance2.append(aux3) # Cosine
    
    distance3 = []
    for i in range(len(distance1)):
        aux = []
        for j in range(len(distance1[0])):
            aux.append(distance1[i][j] + distance2[i][j])
        distance3.append(aux)
    
    
    B = []
    for dist3 in distance3:
        mini = dist3[0]
        mini_idx = 0
        for ii in range(1, len(dist3)):
            if dist3[ii] < mini:
                mini = dist3[ii]
                mini_idx = ii
        B.append(mini_idx)
    
    for i in range(ModelNumber):
        seq = []
        for j,b in enumerate(B):
            if b == i:
                seq.append(j)
        Count.append(len(seq))
        for ii, j in zip(range(min(Count[i],L)), seq):
            Membership[ii,i] = j
            for k in range(W):
                Members[ii,W*i+k] = Uniquesample[j,k]
    
    MemberNumber = Count
    ret_B = np.array(B).reshape(-1,1)
    return Members,MemberNumber,Membership,ret_B 

@njit
def data_standardization_njit(data,X_global,mean_global,mean_global2,k):
    mean_global_new = k/(k+1)*mean_global+data/(k+1)
    X_global_new = k/(k+1)*X_global+np.sum(np.power(data,2))/(k+1)
    mean_global2_new = k/(k+1)*mean_global2+data/(k+1)/np.sqrt(np.sum(np.power(data,2)))
    return X_global_new, mean_global_new, mean_global2_new

@njit
def Chessboard_online_division_njit(data,Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    distance = np.zeros((NB,2))
    COUNT = 0
    SQ = []
    
    W, = BOX_miu[0].shape
    for i in range(NB):     
        aux = 0 # Euclidean
        num = 0 # Cosine
        den1 = 0 # Cosine
        den2 = 0 # Cosine
        for iii in range(W):
            aux += (BOX_miu[i, iii] - data[0, iii])**2 # Euclidean 
            num += BOX_miu[i,iii]*data[0, iii] # Cosine
            den1 += BOX_miu[i,iii]**2 # Cosine
            den2 += data[0, iii]**2 # Cosine
                
        distance[i,0] = aux**.5 # Euclidean
        d2 = (1 - num/(den1**.5 * den2**.5) ) # Cosine
        if d2 < 0:
            distance[i,1] = 0 # Cosine
        else:
            distance[i,1] = d2**.5 # Cosine
                
        
        if distance[i,0] < intervel1 and distance[i,1] < intervel2:
            COUNT += 1
            SQ.append(i)

    L, W = Box.shape
    if COUNT == 0:
        Box_new = np.zeros((L+1, W))
        BOX_miu_new = np.zeros((L+1, W))
        BOX_S_new = np.zeros((L+1), dtype=np.int32)
        for ii in range(L):
            BOX_S_new[ii] = np.int32(BOX_S[ii])
            for jj in range(W):
                Box_new[ii,jj] = Box[ii,jj]
                BOX_miu_new[ii,jj] = BOX_miu[ii,jj]
        for jj in range(W):
            Box_new[L,jj] = data[0, jj]
            BOX_miu_new[L,jj] = data[0, jj]
        
        BOX_S_new[L] = np.int32(1)
        NB_new = NB+1

    if COUNT>=1:
        Box_new = Box
        BOX_S_new = BOX_S
        BOX_miu_new = BOX_miu
        NB_new = NB
        
        DIS = np.zeros((COUNT,1))
        for j in range(COUNT):
            DIS[j] = distance[SQ[j],0] + distance[SQ[j],1]
        
        mini = DIS[0]
        b = 0
        for ii in range(1,len(DIS)):
            if DIS[ii] < mini:
                mini = DIS[ii]
                b = ii
        
        
        BOX_S_new[SQ[b]] += np.int32(1)
        
        for i in range(W):
            BOX_miu_new[SQ[b], i] = BOX_S[SQ[b]]/BOX_S_new[SQ[b]]*BOX_miu[SQ[b], i]+data[0,i]/BOX_S_new[SQ[b]]
    
      
    return Box_new,BOX_miu_new,BOX_S_new,NB_new

@njit
def Chessboard_online_merge_njit(Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    threshold1=intervel1/2
    threshold2=intervel2/2
    NB1=0
    
    L,W = BOX_miu.shape
    deleted_rows = 0
    while NB1 != NB:
        CC = 0
        NB1 = NB
        for ii in range(NB):
            
            seq1 = [i for i in range(NB) if i != ii]
            # distance1 = cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'euclidean')
            
            distance1 = List() # Euclidean
            distance2 = List() # Cosine
            for i in range(NB):
                if i!= ii:
                    aux = 0 # Euclidean
                    num = 0 # Cosine
                    den1 = 0 # Cosine
                    den2 = 0 # Cosine
                    for jj in range(W):
                        aux += (BOX_miu[ii,jj] - BOX_miu[i,jj])**2 # Euclidean
                        num += BOX_miu[ii,jj]*BOX_miu[i,jj] # Cosine
                        den1 += BOX_miu[ii,jj]**2 # Cosine
                        den2 += BOX_miu[i,jj]**2 # Cosine
                    distance1.append(aux**.5) # Euclidean  
                    d2 = (1 - num/(den1**.5 * den2**.5) ) # Cosine
                    if d2 < 0:               
                        distance2.append(d2**.5) # Cosine  
                    else:
                        distance2.append(0) # Cosine               
            
            for jj in range(NB-1):
                if distance1[jj] < threshold1 and distance2[jj] < threshold2:
                    CC = 1
                    NB -= 1
                    #Box = np.delete(Box, (ii))
                    BOX_miu[seq1[jj]] = BOX_miu[seq1[jj]]*BOX_S[seq1[jj]]/(BOX_S[seq1[jj]]+BOX_S[ii])+BOX_miu[ii]*BOX_S[ii]/(BOX_S[seq1[jj]]+BOX_S[ii])
                    
                    BOX_S[seq1[jj]] = BOX_S[seq1[jj]] + BOX_S[ii]
                    
                    
                    ### ----------------------------------------------------------------------- ###
                    
                    
                    #BOX_miu = np.delete(BOX_miu, (ii))
                    #BOX_S = np.delete(BOX_S, (ii))
                    deleted_rows += 1
                    for i in range(L):
                        if i < ii:
                            for iii in range(W):
                                Box[i,iii] = Box[i,iii]
                                BOX_miu[i,iii] = BOX_miu[i,iii]
                            BOX_S[i] = BOX_S[i]
                        elif i < (L - deleted_rows):
                            for iii in range(W):
                                Box[i,iii] = Box[i+1,iii]
                                BOX_miu[i,iii] = BOX_miu[i+1,iii]
                            BOX_S[i] = BOX_S[i+1]
                        else:
                            for iii in range(W):
                                Box[i,iii] = 0
                                BOX_miu[i,iii] = 0
                            BOX_S[i] = 0
                            
                    
                    ### ----------------------------------------------------------------------- ###
                    break
            if CC == 1:
                break        

    if deleted_rows != 0:
        Box_new = Box[:-deleted_rows]
        BOX_miu_new = BOX_miu[:-deleted_rows]
        BOX_S_new = BOX_S[:-deleted_rows]         
        return Box_new,BOX_miu_new,BOX_S_new,NB
    else:        
        return Box,BOX_miu,BOX_S,NB

def Chessboard_globaldensity(Hypermean,HyperSupport,NH):
    uspi1 = pi_calculator(Hypermean,'euclidean')
    sum_uspi1 = np.sum(uspi1)
    Density_1 = uspi1/sum_uspi1
    uspi2 = pi_calculator(Hypermean,'cosine')
    sum_uspi2 = np.sum(uspi2)
    Density_2 = uspi1/sum_uspi2
    Hyper_GD = (Density_2 + Density_1)*HyperSupport
    return Hyper_GD

@njit
def ChessBoard_online_projection_njit(BOX_miu,BOXMT,NB,interval1,interval2):
    Centers = []
    ModeNumber = 0
    n = 2
    W, = BOX_miu[0].shape
    for ii in range(NB):
        Reference = BOX_miu[ii]
        distance1 = np.zeros((NB,1)) # Euclidean
        distance2 = np.zeros((NB,1)) # Cosine
        for i in range(NB):          
            aux = 0 # Euclidean
            num = 0 # Cosine
            den1 = 0 # Cosine
            den2 = 0 # Cosine
            for iii in range(W):
                aux += (Reference[iii] - BOX_miu[i, iii])**2 # Euclidean 
                num += Reference[iii]*BOX_miu[i, iii] # Cosine
                den1 += Reference[iii]**2 # Cosine
                den2 += BOX_miu[i, iii]**2 # Cosine 
            distance1[i] = aux**.5 # Euclidean
            d2 = (1 - num/(den1**.5 * den2**.5) ) # Cosine
            if d2 < 0:
                distance2[i] = 0 # Cosine
            else:
                distance2[i] = d2**.5 # Cosine
        
        Chessblocak_typicality = []
        for i in range(NB):
            if distance1[i]<n*interval1 and distance2[i]<n*interval2:
                Chessblocak_typicality.append(BOXMT[i])
        if max(Chessblocak_typicality) == BOXMT[ii]:
            Centers.append(Reference)
            ModeNumber += 1
    return Centers,ModeNumber

def SelfOrganisedDirectionAwareDataPartitioning(Input, Mode):
    
    if Mode == 'Offline':
        data = Input['StaticData']
        L, W = data.shape
        N = Input['GridSize']
        distancetype = Input['DistanceType']

        #print(N, '-', datetime.now())
        X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)

        GD, Uniquesample, Frequency = Globaldensity_Calculator(data, distancetype)

        BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division_njit(Uniquesample,GD,grid_trad,grid_angl, distancetype)
        BOX = np.asarray(BOX)
        BOX_miu = np.asarray(BOX_miu)
        BOX_S = np.asarray(BOX_S)

        Center,ModeNumber = ChessBoard_PeakIdentification_njit(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
        
        Center_numba = List(Center)
        Members,Membernumber,Membership,IDX = cloud_member_recruitment_njit(ModeNumber,Center_numba,data,grid_trad,float(grid_angl), distancetype)
        
        Boxparameter = {'BOX': BOX,
                'BOX_miu': BOX_miu,
                'BOX_S': BOX_S,
                'NB': NB,
                'XM': X1,
                'L': L,
                'AvM': AvD1,
                'AvA': AvD2,
                'GridSize': N}
        

    if Mode == 'Evolving':
        distancetype = Input['DistanceType']
        Data2 = Input['StreamingData']
        data = Input['AllData']
        Boxparameter = Input['SystemParams']
        BOX = Boxparameter['BOX']
        BOX_miu = Boxparameter['BOX_miu']
        BOX_S = Boxparameter['BOX_S']
        XM = Boxparameter['XM']
        AvM = Boxparameter['AvM']
        AvA = Boxparameter ['AvA']
        N = Boxparameter ['GridSize']
        NB = Boxparameter ['NB']
        L1 = Boxparameter ['L']
        L2, _ = Data2.shape

        for k in range(L2):
            XM, AvM, AvA = data_standardization_njit(Data2[k,:], XM, AvM, AvA, k+L1)

            interval1 = np.sqrt(2*(XM-np.sum(np.power(AvM,2))))/N
            interval2 = np.sqrt(1-np.sum(np.power(AvA,2)))/N
            
            BOX, BOX_miu, BOX_S, NB = Chessboard_online_division_njit(np.array(Data2[k,:]), BOX, BOX_miu, BOX_S, NB, interval1, interval2)

            BOX,BOX_miu,BOX_S,NB = Chessboard_online_merge_njit(BOX,BOX_miu,BOX_S,NB,interval1,interval2)

        BOXG = Chessboard_globaldensity(BOX_miu,BOX_S,NB)

        Center, ModeNumber = ChessBoard_online_projection_njit(BOX_miu,BOXG,NB,interval1,interval2)

        Center_numba = List(Center)
        Members, Membernumber, _, IDX = cloud_member_recruitment_njit(ModeNumber, Center_numba, data, interval1, interval2, distancetype)
        
        
        Boxparameter['BOX']=BOX
        Boxparameter['BOX_miu']=BOX_miu
        Boxparameter['BOX_S']=BOX_S
        Boxparameter['NB']=NB
        Boxparameter['L']=L1+L2
        Boxparameter['AvM']=AvM
        Boxparameter['AvA']=AvA
        

    Output = {'C': Center,
              'IDX': IDX,
              'SystemParams': Boxparameter,
              'DistanceType': distancetype}
    return Output
