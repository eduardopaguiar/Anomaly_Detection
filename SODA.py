from scipy.spatial.distance import pdist, cdist, squareform
import math
import pandas as pd
import numpy as np
import cupy as cp

def argwhere(data):
    aux1 = cp.nonzero(data)
    aux2 = aux1[0].reshape(-1,1)
    
    for i in range (1,len(aux1)):
        aux2 = cp.concatenate((aux1[i].reshape(-1,1),aux2), axis = 1)
    
    return aux2

def cp_cdist(A, B, metric='euclidean'):
    LA, WA = A.shape
    LB, WB = B.shape
    ret_array = cp.zeros((LA,LB))
    
    if metric == 'euclidean':
        aux = cp.empty((LB,WB))
        
        for i in range(LA):
            for j in range(LB):
                aux[j] = A[i]-B[j]
            ret_array[i] = cp.sqrt(cp.sum(cp.power(aux,2),axis=1))
    
    if metric == 'cosine':
        num = cp.empty((LB))
        den = cp.empty((LB))
        
        for i in range(LA):
            for j in range(LB):
                num [j]= cp.sum(A[i]*B[j])
                den [j]= cp.sqrt(cp.sum(A[i]*A[i])*cp.sum(B[j]*B[j]))
            ret_array[i] = 1 - num/den
    
    return ret_array

def cp_pdist(A, metric='euclidean'):
    L, WA = A.shape
    
    if metric == 'euclidean':
        for i in range(L):
            aux = cp.empty(((L-(i+1)),WA))
            for j in range(i+1, L):
                aux[j-(i+1)]= (A[i] - A[j])
            if i == 0:
                ret_array = cp.sqrt(cp.sum(cp.power(aux,2),axis=1))
            elif i == L-1:
                break
            else:
                ret_array = cp.concatenate((ret_array, cp.sqrt(cp.sum(cp.power(aux,2),axis=1))))
                
    if metric == 'cosine':
        for i in range(L):
            num = cp.empty(((L-(i+1))))
            den = cp.empty(((L-(i+1))))
            for j in range(i+1, L):
                num[j-(i+1)] = cp.sum(A[i]*A[j])
                den[j-(i+1)] = cp.sqrt(cp.sum(A[i]*A[i])*cp.sum(A[j]*A[j]))
            if i == 0:
                ret_array = 1 - num/den
            elif i == L-1:
                break
            else:
                ret_array = cp.concatenate((ret_array, 1 - num/den))
       
    return ret_array

def cp_pdist_squareform(A, metric='euclidean'):
    L, WA = A.shape
    
    if metric == 'euclidean':
        ret_array = cp.zeros((L,L))
    
        for i in range(L):
            aux = cp.empty(((L-(i+1)),WA))
            for j in range(i+1, L):
                aux[j-(i+1)] = A[i] - A[j]
            if i == L-1:
                break
            else:
                ret_array[i, (i+1):L] = cp.sqrt(cp.sum(cp.power(aux,2),axis=1))
                ret_array[(i+1):L, i] = cp.sqrt(cp.sum(cp.power(aux,2),axis=1))
        
    if metric == 'cosine':
        ret_array = cp.zeros((L,L))
    
        for i in range(L):
            num = cp.empty(((L-(i+1))))
            den = cp.empty(((L-(i+1))))
            for j in range(i+1, L):
                num[j-(i+1)] = cp.sum(A[i]*A[j])
                den[j-(i+1)] = cp.sqrt(cp.sum(A[i]*A[i])*cp.sum(A[j]*A[j]))
            if i == L-1:
                break
            else:
                ret_array[i, (i+1):L] = 1 - num/den
                ret_array[(i+1):L, i] = 1 - num/den
            
    return ret_array
  
def grid_set(data, N):

    _ , W = data.shape
    AvD1 = data.mean(0)
    X1 = cp.mean(cp.sum(cp.power(data,2),axis=1))
    grid_trad = cp.sqrt(2*(X1 - cp.sum(AvD1*AvD1.T)))/N
    Xnorm = cp.sqrt(cp.sum(cp.power(data,2),axis=1)).reshape(-1,1)
    aux = Xnorm
    for i in range(W-1):
        aux = cp.concatenate((Xnorm,aux),axis=1)
        #aux = cp.insert(aux,0,Xnorm.T,axis=1)
    data = data / aux
    seq = argwhere(cp.isnan(data))
    if tuple(seq[::]): data[tuple(seq[::])] = 1
    AvD2 = data.mean(0)
    grid_angl = cp.sqrt(1-cp.sum(AvD2*AvD2.T))/N
    return X1, AvD1, AvD2, grid_trad, grid_angl

def pi_calculator(Uniquesample, mode):
    UN, W = Uniquesample.shape
    if mode == 'euclidean':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(cp.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(cp.power(AA1,2))
        aux = []
        aux2 = cp.empty((Uniquesample.shape))
        for i in range(UN): aux.append(AA1)
        for i in range(UN): aux2 [i] = Uniquesample[i]-aux[i]
        a = cp.power(aux2,2)
        b = cp.sum(a,axis=1)
        uspi = b+DT1

    
    if mode == 'cosine':
        #Xnorm = cp.matrix(cp.sqrt(cp.sum(cp.power(Uniquesample,2),axis=1))).T
        Xnorm = cp.sqrt(cp.sum(cp.power(Uniquesample,2),axis=1)).reshape(-1,1).T
        aux2 = Xnorm.T
        for i in range(W-1):
            aux2 = cp.concatenate((Xnorm.T,aux2),axis=1)
            
        #aux2 = cp.insert(aux2,0,Xnorm.T,axis=1)
        Uniquesample1 = Uniquesample / aux2
        AA2 = cp.mean(Uniquesample1,0).reshape(1,-1)
        X2 = 1
        DT2 = X2 - cp.sum(cp.power(AA2,2))
        aux = []
        aux2 = cp.empty((UN,1,W))
        for i in range(UN): aux.append(AA2)
        
        for i in range(UN): aux2 [i] = Uniquesample1[i]-aux[i]
        
        a = cp.power(aux2,2)
        b = cp.sum(a,axis=1)
        c = cp.sum(b,axis=1)
        uspi = c+DT2
        
    return uspi

def Globaldensity_Calculator(data, distancetype):
    data = cp.asnumpy(data)
    Uniquesample, J, K = np.unique(data, axis=0, return_index=True, return_inverse=True)
    Uniquesample =  cp.asarray(Uniquesample)
    Frequency, _ = np.histogram(K,bins=len(J))
    Frequency = cp.asarray(Frequency)
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

def chessboard_division(Uniquesample, MMtypicality, interval1, interval2, distancetype):
    L, WU = Uniquesample.shape
    W = 1
    BOX = cp.asarray(Uniquesample[0]).reshape(1,-1)
    BOX_miu = cp.asarray(Uniquesample[0]).reshape(1,-1)
    BOX_S = cp.asarray([1])
    BOX_X = cp.sum(cp.power(cp.asarray(Uniquesample[0]),2)).reshape(1,-1)
    NB = W
    BOXMT = cp.asarray(MMtypicality[0]).reshape(1,-1)
    
    for i in range(W,L):
        a = cp_cdist(Uniquesample[i].reshape(1,-1), BOX_miu)
        b = cp.sqrt(cp_cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric='cosine'))
        distance = cp.stack((a[0],b[0])).T
        SQ = []
        for j,d in enumerate(distance):
            if d[0] < interval1 and d[1] < interval2:
                SQ.append(j)
        #SQ = cp.argwhere(distance[::,0]<interval1 and (distance[::,1]<interval2))
        COUNT = len(SQ)
        if COUNT == 0:
            BOX = cp.vstack((BOX, cp.asarray(Uniquesample[i]).reshape(1,-1)))
            NB = NB + 1
            BOX_S = cp.vstack((BOX_S, cp.asarray([1])))
            BOX_miu = cp.vstack((BOX_miu, cp.asarray(Uniquesample[i]).reshape(1,-1)))
            BOX_X = cp.vstack((BOX_X, cp.asarray(cp.sum(cp.power(cp.asarray(Uniquesample[0]).reshape(1,-1),2))).reshape(1,-1)))
            BOXMT = cp.vstack((BOXMT, cp.asarray(MMtypicality[i]).reshape(1,-1)))
        if COUNT >= 1:
            DIS = distance[SQ[::],0]/interval1 + distance[SQ[::],1]/interval2 # pylint: disable=E1136  # pylint/issues/3139
            b = cp.argmin(DIS)
            BOX_S[SQ[int(b)]] = BOX_S[SQ[int(b)]] + 1
            BOX_miu[SQ[int(b)]] = (BOX_S[SQ[int(b)]]-1)/BOX_S[SQ[int(b)]]*BOX_miu[SQ[int(b)]] + Uniquesample[i]/BOX_S[SQ[int(b)]]
            BOX_X[SQ[int(b)]] = (BOX_S[SQ[int(b)]]-1)/BOX_S[SQ[int(b)]]*BOX_X[SQ[int(b)]] + sum(Uniquesample[i]**2)/BOX_S[SQ[int(b)]]
            BOXMT[SQ[int(b)]] = BOXMT[SQ[int(b)]] + MMtypicality[i]
    return BOX, BOX_miu, BOX_X, BOX_S, BOXMT, NB

def ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,Internval1,Internval2, distancetype):
    Centers = []
    n = 2
    ModeNumber = 0
    
    distance1 = cp_pdist_squareform(BOX_miu,metric=distancetype)
    
    distance2 = cp.sqrt(cp_pdist_squareform(BOX_miu,metric='cosine'))
    for i in range(NB):
        seq = []
        for j,(d1,d2) in enumerate(zip(distance1[i],distance2[i])):
            if d1 < n*Internval1 and d2 < n*Internval2:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]
        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1
    Centers_array = cp.empty((len(Centers),len(Centers[0])))
    for i in range(len(Centers)): Centers_array[i] = Centers[i]

    return Centers_array, ModeNumber

def cloud_member_recruitment(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    L, W = Uniquesample.shape
    Membership = cp.zeros((L,ModelNumber))
    Members = cp.zeros((L,ModelNumber*W))
    Count = []

    distance1 = cp_cdist(Uniquesample,Center_samples, metric=distancetype)/grid_trad

    distance2 = cp.sqrt(cp_cdist(Uniquesample, Center_samples, metric='cosine'))/grid_angl
    distance3 = distance1 + distance2
    B = distance3.argmin(1)
    for i in range(ModelNumber):
        seq = []
        for j,b in enumerate(B):
            if b == i:
                seq.append(j)
        Count.append(len(seq))
        for j in range(len(seq)): Membership[(j),i] = seq[j]
        aux = cp.empty((len(seq),W))
        k = 0
        for j in seq:
            aux[k] = Uniquesample[j]
            k += 1
        for j in range(len(seq)): Members[j,W*i:W*(i+1)] = aux[j]
    MemberNumber = Count
    return Members,MemberNumber,Membership,B 

def data_standardization(data,X_global,mean_global,mean_global2,k):
    mean_global_new = k/(k+1)*mean_global+data/(k+1)
    X_global_new = k/(k+1)*X_global+cp.sum(cp.power(data,2))/(k+1)
    mean_global2_new = k/(k+1)*mean_global2+data/(k+1)/cp.sqrt(cp.sum(cp.power(data,2)))
    return X_global_new, mean_global_new, mean_global2_new

def Chessboard_online_division(data,Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    distance = cp.zeros((NB,2))
    COUNT = 0
    SQ = []
    aux = cp.empty((2,len(BOX_miu[0])))
    for i in range(NB):
        aux = cp.stack([BOX_miu[i], data])
        distance[i,0] = float(cp_pdist(aux,'euclidean'))
        distance[i,1] = float(cp.sqrt(cp_pdist(aux,'cosine')))
        if distance[i,0] < intervel1 and distance[i,1] < intervel2:
            COUNT += 1
            SQ.append(i)
            
    if COUNT == 0:
        Box_new = cp.concatenate((Box, data.reshape(1,-1)))
        NB_new = NB+1
        BOX_S_new = cp.concatenate((BOX_S, cp.asarray([1]).reshape(1,-1)))
        #BOX_S_new = cp.array(BOX_S)
        BOX_miu_new = cp.concatenate((BOX_miu, cp.array(data.reshape(1,-1))))
    if COUNT>=1:
        DIS = cp.zeros((COUNT,1))
        for j in range(COUNT):
            DIS[j] = distance[SQ[j],0] + distance[SQ[j],1]
        a = cp.amin(DIS)
        b = int(cp.where(DIS == a)[0])
        Box_new = Box
        NB_new = NB
        BOX_S_new = cp.array(BOX_S)
        BOX_miu_new = cp.array(BOX_miu)
        BOX_S_new[SQ[b]] = BOX_S[SQ[b]] + 1
        BOX_miu_new[SQ[b]] = BOX_S[SQ[b]]/BOX_S_new[SQ[b]]*BOX_miu[SQ[b]]+data/BOX_S_new[SQ[b]]
    
            
    
    return Box_new,BOX_miu_new,BOX_S_new,NB_new

def Chessboard_online_merge(Box,BOX_miu,BOX_S,NB,intervel1,intervel2):
    threshold1=intervel1/2
    threshold2=intervel2/2
    NB1=0
    
    while NB1 != NB:
        CC = 0
        NB1 = NB
        for ii in range(NB):
            seq1 = [i for i in range(NB) if i != ii]
            distance1 = cp_cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'euclidean')
            distance2 = cp.sqrt(cp_cdist(BOX_miu[ii].reshape(1,-1), BOX_miu[seq1], 'cosine'))
            for jj in range(NB-1):
                if distance1[0,jj] < threshold1 and distance2[0,jj] < threshold2:
                    CC = 1
                    NB -= 1
                    #Box = np.delete(Box, (ii), axis=0)
                    BOX_miu[seq1[jj]] = BOX_miu[seq1[jj]]*BOX_S[seq1[jj]]/(BOX_S[seq1[jj]]+BOX_S[ii])+BOX_miu[ii]*BOX_S[ii]/(BOX_S[seq1[jj]]+BOX_S[ii])
                    
                    BOX_S[seq1[jj]] = BOX_S[seq1[jj]] + BOX_S[ii]
                    #BOX_miu = np.delete(BOX_miu, (ii), axis=0)
                    #BOX_S = np.delete(BOX_S, (ii), axis=0)
    
                    L, W1 = Box.shape
                    W2 = BOX_miu.shape[1]
                    W3 = BOX_S.shape[1]
                    new_Box = cp.empty((L-1,W1))
                    new_BOX_miu = cp.empty((L-1, W2))
                    new_BOX_S = cp.empty((L-1,W3))
                    for ll, kk in enumerate([xx for xx in range(L) if xx != ii]):
                        new_Box[ll] = Box[kk]
                        new_BOX_miu[ll] = BOX_miu[kk]
                        new_BOX_S[ll] = BOX_S[kk]
                    Box = new_Box
                    BOX_miu = new_BOX_miu
                    BOX_S = new_BOX_S
                    break
            if CC == 1:
                break
                    
    return Box,BOX_miu,BOX_S,NB

def Chessboard_globaldensity(Hypermean,HyperSupport,NH):
    uspi1 = pi_calculator(Hypermean,'euclidean')
    sum_uspi1 = cp.sum(uspi1)
    Density_1 = uspi1/sum_uspi1
    uspi2 = pi_calculator(Hypermean,'cosine')
    sum_uspi2 = cp.sum(uspi2)
    Density_2 = uspi1/sum_uspi2
    Hyper_GD = (Density_2 + Density_1)*HyperSupport.T
    return Hyper_GD

def ChessBoard_online_projection(BOX_miu,BOXMT,NB,interval1,interval2):
    Centers = []
    ModeNumber = 0
    n = 2
    
    for ii in range(NB):
        Reference = BOX_miu[ii]
        distance1 = cp.zeros((NB,1))
        distance2 = cp.zeros((NB,1))
        for i in range(NB):
            distance1[i] = cp_pdist(cp.vstack((Reference, BOX_miu[i])), 'euclidean')
            distance2[i] = cp.sqrt(cp_pdist(cp.vstack((Reference, BOX_miu[i])), 'cosine'))
        
        first = True
        for i in range(NB):
            if distance1[i]<n*interval1 and distance2[i]<n*interval2:
                if first:
                    Chessblocak_typicality = BOXMT[0][i].reshape(1,-1)
                    first = False
                else:
                    Chessblocak_typicality = cp.vstack((Chessblocak_typicality, BOXMT[0][i].reshape(1,-1)))
                    
        if max(Chessblocak_typicality) == BOXMT[0][ii]:
            Centers.append(Reference)
            ModeNumber += 1
    Centers = cp.asarray(Centers)
    return Centers,ModeNumber

def SelfOrganisedDirectionAwareDataPartitioning(Input, Mode):
    if Mode == 'Offline':
        data = Input['StaticData']
        L, W = data.shape
        N = Input['GridSize']
        distancetype = Input['DistanceType']
        X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)
        GD, Uniquesample, Frequency = Globaldensity_Calculator(data, distancetype)
        BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division(Uniquesample,GD,grid_trad,grid_angl, distancetype)
        Center,ModeNumber = ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
        Members,Membernumber,Membership,IDX = cloud_member_recruitment(ModeNumber,Center,data,grid_trad,grid_angl, distancetype)
        
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
            interval1 = cp.sqrt(2*(XM-cp.sum(cp.power(AvM,2))))/N
            interval2 = cp.sqrt(1-cp.sum(cp.power(AvA,2)))/N
            BOX, BOX_miu, BOX_S, NB = Chessboard_online_division(Data2[k,:], BOX, BOX_miu, BOX_S, NB, interval1, interval2)
            BOX,BOX_miu,BOX_S,NB = Chessboard_online_merge(BOX,BOX_miu,BOX_S,NB,interval1,interval2)
        
        BOXG = Chessboard_globaldensity(BOX_miu,BOX_S,NB)
        Center, ModeNumber = ChessBoard_online_projection(BOX_miu,BOXG,NB,interval1,interval2)
        Members, Membernumber, _, IDX = cloud_member_recruitment(ModeNumber, Center, data, interval1, interval2, distancetype)
        
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
