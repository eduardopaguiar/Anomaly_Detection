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
    BOXMT = [MMtypicality[k] for k in range(W)]

    for i in range(W,L):
        XA = Uniquesample[i].reshape(1,-1)
        XB = BOX_miu
        a = []
        for ii in range (len(XA)):
            aux2 = []
            for j in range (len(XB)):
                aux1 = []
                bux1 = 0 
                for e1, e2 in zip(XB[j],XA[ii,:]):
                    aux1.append((e1-e2)**2)
                    bux1 += ((e1-e2)**2)
                aux2.append(bux1**(0.5))
            a.append(aux2)
            
        b = []
        for ii in range(len(XA)):
            aux1 = []
            for j in range(len(XB)):
                dot = 0
                denom_a = 0
                denom_b = 0
                for k in range (len(XB[j])):
                    dot += (XB[j][k]*XA[ii,k])
                    denom_a += (XB[j][k] * XB[j][k])
                    denom_b += (XA[ii,k] * XA[ii,k])

                aux1.append((1 - ((dot / ((denom_a ** 0.5) * (denom_b ** 0.5)))))**0.5)
            b.append(aux1)
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
def cloud_member_recruitment_njit(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    L, W = Uniquesample.shape
    Membership = np.zeros((L,ModelNumber))
    Members = np.zeros((L,ModelNumber*W))
    Count = []
    
    distance1 = []
    for ii in range(L):
        aux2 = []
        for j in range (len(Center_samples)):
            aux1 = []
            bux1 = 0 
            for k in range(W):
                aux1.append((Center_samples[j][k]-Uniquesample[ii,k])**2)
                bux1 += ((Center_samples[j][k]-Uniquesample[ii,k])**2)
            aux2.append((bux1**(0.5))/grid_trad)
        distance1.append(aux2)

    
    distance2 = []
    for ii in range(L):
        aux1 = []
        for j in range(len(Center_samples)):
            dot = 0
            denom_a = 0
            denom_b = 0
            for k in range (len(Center_samples[j])):
                dot += (Center_samples[j][k]*Uniquesample[ii,k])
                denom_a += (Center_samples[j][k] * Center_samples[j][k])
                denom_b += (Uniquesample[ii,k] * Uniquesample[ii,k])

            aux1.append(((1 - ((dot / ((denom_a ** 0.5) * (denom_b ** 0.5)))))**0.5)/grid_angl)
        distance2.append(aux1)
    
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

def cloud_member_recruitment_std(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    L, W = Uniquesample.shape
    Membership = np.zeros((L,ModelNumber))
    Members = np.zeros((L,ModelNumber*W))
    Count = []



    if distancetype == 'minkowski':
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype, p=1.5)/grid_trad
    else:
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype)/grid_trad

    distance2 = np.sqrt(cdist(Uniquesample, Center_samples, metric='cosine'))/grid_angl

    distance3 = distance1 + distance2
    B = distance3.argmin(1)
    for i in range(ModelNumber):
        seq = []
        for j,b in enumerate(B):
            if b == i:
                seq.append(j)
        Count.append(len(seq))
        Membership[:Count[i]:,i] = seq
        Members[:Count[i]:,W*i:W*(i+1)] = [Uniquesample[j] for j in seq]
    MemberNumber = Count
    return Members,MemberNumber,Membership,B 

def data_standardization(data,X_global,mean_global,mean_global2,k):
    mean_global_new = k/(k+1)*mean_global+data/(k+1)
    X_global_new = k/(k+1)*X_global+np.sum(np.power(data,2))/(k+1)
    mean_global2_new = k/(k+1)*mean_global2+data/(k+1)/np.sqrt(np.sum(np.power(data,2)))
    return X_global_new, mean_global_new, mean_global2_new

def Chessboard_online_division_std(data,Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    distance = np.zeros((NB,2))
    COUNT = 0
    SQ = []

    for i in range(NB):


        distance[i,0] = pdist([list(BOX_miu[i]), data.tolist()[0]],'euclidean')

        distance[i,1] = np.sqrt(pdist([list(BOX_miu[i]), data.tolist()[0]],'cosine'))
        
        if distance[i,0] < intervel1 and distance[i,1] < intervel2:
            COUNT += 1
            SQ.append(i)

    if COUNT == 0:
        Box_new = np.concatenate((Box, np.array(data)))
        NB_new = NB+1
        BOX_S_new = np.concatenate((BOX_S, np.array([1])))
        #BOX_S_new = np.array(BOX_S)
        BOX_miu_new = np.concatenate((BOX_miu, np.array(data)))
    if COUNT>=1:
        DIS = np.zeros((COUNT,1))
        for j in range(COUNT):
            DIS[j] = distance[SQ[j],0] + distance[SQ[j],1]
        a = np.amin(DIS)
        b = int(np.where(DIS == a)[0])
        Box_new = Box
        NB_new = NB
        BOX_S_new = np.array(BOX_S)
        BOX_miu_new = np.array(BOX_miu)
        BOX_S_new[SQ[b]] = BOX_S[SQ[b]] + 1
        BOX_miu_new[SQ[b]] = BOX_S[SQ[b]]/BOX_S_new[SQ[b]]*BOX_miu[SQ[b]]+data/BOX_S_new[SQ[b]]
    
            
    
    return Box_new,BOX_miu_new,BOX_S_new,NB_new

@njit
def Chessboard_online_division_njit(data,Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    distance = np.zeros((NB,2))
    COUNT = 0
    SQ = []
    
    
    W, = BOX_miu[0].shape
    for i in range(NB):
        # pdist euclidean
        # distance[i,0] = pdist([list(BOX_miu[i]), data.tolist()[0]],'euclidean')
            
        aux = 0
        for iii in range(W):
            aux += (BOX_miu[i, iii] - data[0, iii])**2 
                
        distance[i,0] = aux**.5
                
        # pdist cosine
        # distance[i,1] = np.sqrt(pdist([list(BOX_miu[i]), data.tolist()[0]],'cosine'))
            
        num = 0
        den1 = 0
        den2 = 0
        for iii in range(W):
            num += BOX_miu[i,iii]*data[0, iii]
            den1 += BOX_miu[i,iii]**2
            den2 += data[0, iii]**2
            
        d2 = (1 - num/(den1**.5 * den2**.5) )
        if d2 < 0:
            distance[i,1] = 0
        else:
            distance[i,1] = d2**.5
                
        
        if distance[i,0] < intervel1 and distance[i,1] < intervel2:
            COUNT += 1
            SQ.append(i)

            
    Box_new = List(Box)
    BOX_S_new = List(BOX_S)
    BOX_miu_new = List(BOX_miu)
    
    if COUNT == 0:
        Box_new.append(List(data)[0])
        NB_new = NB+1
        BOX_S_new.append(1)
        BOX_miu_new.append(List(data)[0])
        
    if COUNT>=1:
        DIS = np.zeros((COUNT,1))
        for j in range(COUNT):
            DIS[j] = distance[SQ[j],0] + distance[SQ[j],1]
        
        mini = DIS[0]
        for ii in range(1,len(DIS)):
            if DIS[ii] < mini:
                mini = DIS[ii]
                b = ii

        
        NB_new = NB
        BOX_S_new[SQ[b]] = BOX_S[SQ[b]] + 1
        for i in range(W):
            BOX_miu_new[SQ[b]][i] = BOX_S[SQ[b]]/BOX_S_new[SQ[b]]*BOX_miu[SQ[b]][i]+data[0,i]/BOX_S_new[SQ[b]]
    
    
    #Box_new = np.asarray(Box_new)
    #BOX_miu_new = np.asarray(BOX_miu_new)
    #BOX_S_new = np.asarray(BOX_S_new)
    return Box_new,BOX_miu_new,BOX_S_new,NB_new

def Chessboard_online_merge_std(Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    threshold1=intervel1/2
    threshold2=intervel2/2
    NB1=0
    
    while NB1 != NB:
        CC = 0
        NB1 = NB
        for ii in range(NB):
            seq1 = [i for i in range(NB) if i != ii]

            distance1 = cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'euclidean')

            distance2 = np.sqrt(cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'cosine'))

            for jj in range(NB-1):
                if distance1[0,jj] < threshold1 and distance2[0,jj] < threshold2:
                    CC = 1
                    NB -= 1
                    Box = np.delete(Box, (ii), axis=0)
                    BOX_miu[seq1[jj]] = BOX_miu[seq1[jj]]*BOX_S[seq1[jj]]/(BOX_S[seq1[jj]]+BOX_S[ii])+BOX_miu[ii]*BOX_S[ii]/(BOX_S[seq1[jj]]+BOX_S[ii])
                    
                    BOX_S[seq1[jj]] = BOX_S[seq1[jj]] + BOX_S[ii]
                    BOX_miu = np.delete(BOX_miu, (ii), axis=0)
                    BOX_S = np.delete(BOX_S, (ii), axis=0)
                    
                    break
            if CC == 1:
                break                 
    return Box,BOX_miu,BOX_S,NB

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
            
            distance1 = List()
            for i in range(NB):
                if i!= ii:
                    aux = 0
                    for jj in range(W):
                        aux += (BOX_miu[ii,jj] - BOX_miu[i,jj])**2
                
                    distance1.append(aux**.5)  
                        
                        
            #distance2 = np.sqrt(cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'cosine'))
            
            distance2 = List()
            for i in range(NB):
                if i!= ii:
                    num = 0
                    den1 = 0
                    den2 = 0
                    for jj in range(W):
                        num += BOX_miu[ii,jj]*BOX_miu[i,jj]
                        den1 += BOX_miu[ii,jj]**2
                        den2 += BOX_miu[i,jj]**2
                    
                    d2 = (1 - num/(den1**.5 * den2**.5) )
                    if d2 < 0:               
                        distance2.append(d2**.5)  
                    else:
                        distance2.append(0)
            
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
    return Box,BOX_miu,BOX_S,NB, deleted_rows

def Chessboard_globaldensity(Hypermean,HyperSupport,NH):
    uspi1 = pi_calculator(Hypermean,'euclidean')
    sum_uspi1 = np.sum(uspi1)
    Density_1 = uspi1/sum_uspi1
    uspi2 = pi_calculator(Hypermean,'cosine')
    sum_uspi2 = np.sum(uspi2)
    Density_2 = uspi1/sum_uspi2
    Hyper_GD = (Density_2 + Density_1)*HyperSupport
    return Hyper_GD

def ChessBoard_online_projection_std(BOX_miu,BOXMT,NB,interval1,interval2):
    Centers = []
    ModeNumber = 0
    n = 2
    
    for ii in range(NB):
        Reference = BOX_miu[ii]
        distance1 = np.zeros((NB,1))
        distance2 = np.zeros((NB,1))
        for i in range(NB):
            distance1[i] = pdist([list(Reference), list(BOX_miu[i])], 'euclidean')

            distance2[i] = np.sqrt(pdist([list(Reference), list(BOX_miu[i])], 'cosine'))
            
        
        Chessblocak_typicality = []
        for i in range(NB):
            if distance1[i]<n*interval1 and distance2[i]<n*interval2:
                Chessblocak_typicality.append(BOXMT[i])
        if max(Chessblocak_typicality) == BOXMT[ii]:
            Centers.append(Reference)
            ModeNumber += 1
    return Centers,ModeNumber

@njit
def ChessBoard_online_projection_njit(BOX_miu,BOXMT,NB,interval1,interval2):
    Centers = []
    ModeNumber = 0
    n = 2
    
    for ii in range(NB):
        Reference = BOX_miu[ii]
        distance1 = np.zeros((NB,1))
        distance2 = np.zeros((NB,1))
        for i in range(NB):
            
            
            # pdist euclidean
            # distance1[i] = pdist(np.vstack((Reference, BOX_miu[i])), 'euclidean')
            W, = Reference.shape
            
            aux = 0
            for iii in range(W):
                aux += (Reference[iii] - BOX_miu[i, iii])**2 
                
            distance1[i] = aux**.5
                
            # pdist cosine
            # distance2[i] = np.sqrt(pdist(np.vstack((Reference, BOX_miu[i])), 'cosine'))
            
            
            num = 0
            den1 = 0
            den2 = 0
            for iii in range(W):
                num += Reference[iii]*BOX_miu[i, iii]
                den1 += Reference[iii]**2
                den2 += BOX_miu[i, iii]**2
            
            d2 = (1 - num/(den1**.5 * den2**.5) )
            if d2 < 0:
                distance2[i] = 0
            else:
                distance2[i] = d2**.5
              
        
        Chessblocak_typicality = []
        for i in range(NB):
            if distance1[i]<n*interval1 and distance2[i]<n*interval2:
                Chessblocak_typicality.append(BOXMT[i])
        if max(Chessblocak_typicality) == BOXMT[ii]:
            Centers.append(Reference)
            ModeNumber += 1
    return Centers,ModeNumber

def SelfOrganisedDirectionAwareDataPartitioning(Input, Mode):
    
    execution_time = open('/AtlasDisk/user/pestana/Output/.csv', 'a+')
    if Mode == 'Offline':
        data = Input['StaticData']
        L, W = data.shape
        N = Input['GridSize']
        distancetype = Input['DistanceType']

        X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)
        GD, Uniquesample, Frequency = Globaldensity_Calculator(data, distancetype)
        BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division_njit(Uniquesample,GD,grid_trad,grid_angl, distancetype)
        BOX = np.asarray(BOX)
        BOX_miu = np.asarray(BOX_miu)
        BOX_S = np.asarray(BOX_S)
        Center,ModeNumber = ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
        Center_numba = List(Center)
        Members,Membernumber,Membership,IDX = cloud_member_recruitment_njit(ModeNumber,Center_numba,data,grid_trad,grid_angl, distancetype)

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
            XM, AvM, AvA = data_standardization(Data2[k,:], XM, AvM, AvA, k+L1)
            interval1 = np.sqrt(2*(XM-np.sum(np.power(AvM,2))))/N
            interval2 = np.sqrt(1-np.sum(np.power(AvA,2)))/N

            
            BOX, BOX_miu, BOX_S, NB = Chessboard_online_division_njit(np.array(Data2[k,:]), BOX, BOX_miu, BOX_S, NB, interval1, interval2)
            BOX = np.asarray(BOX)
            BOX_miu = np.asarray(BOX_miu)
            BOX_S = np.asarray(BOX_S)
            BOX,BOX_miu,BOX_S,NB, deleted_rows = Chessboard_online_merge_njit(BOX,BOX_miu,BOX_S,NB,interval1,interval2)
            if deleted_rows != 0:
                BOX = BOX[:-deleted_rows]
                BOX_miu = BOX_miu[:-deleted_rows]
                BOX_S = BOX_S[:-deleted_rows]
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
        

    execution_time.close()
    Output = {'C': Center,
              'IDX': IDX,
              'SystemParams': Boxparameter,
              'DistanceType': distancetype}
    return Output
