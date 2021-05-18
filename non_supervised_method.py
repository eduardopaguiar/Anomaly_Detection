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

if __name__ == '__main__':
    ### Make detections with the scikit methods

    # Example settings
    signal_fraction = 0.01

    names = ["One-Class SVM", "Isolation Forest", "Local Outlier Factor"]
    anomaly_algorithms = [svm.OneClassSVM(nu=signal_fraction, kernel="rbf"
                        , gamma=0.1),
                        IsolationForest(contamination=signal_fraction, random_state=42),
                        LocalOutlierFactor(n_neighbors=3, contamination=signal_fraction)]

    for it in range(10):
        
        print('\n     => Iteration Number', (it+1) )

        with open('kernel/data_clouds_dic_iteration_' + str(it) + '.pkl', 'rb') as fp:
            data_clouds_dic = pickle.load(fp)

        accuracy_dict = {}
        for gra in data_clouds_dic:
            print('\n           => Granularity', gra )

            aux = {}
            for dc in data_clouds_dic[gra]:
                X = data_clouds_dic[gra][dc]['data']
                Y = data_clouds_dic[gra][dc]['target']
                models_dict = {}
                for name, algorithm in zip(names,anomaly_algorithms):
                    aux2 = {}
                    t0 = time.time()

                    if np.shape(X)[0] != 1:

                        y_pred = algorithm.fit_predict(X)
                        tn, fp, fn, tp = confusion_matrix(Y, y_pred, labels=[1,-1]).ravel()
                        t1 = time.time()    

                        aux2["True_Positive"] = tp
                        aux2["True_Negative"] = tn
                        aux2["False_Positive"] = fp
                        aux2["False_Negative"] = fn
                        aux2["Time"] = t1-t0
                        models_dict[name] = aux2                       

                    else:
                        models_dict[name] = 'mono_event'

                aux[dc] = models_dict
            accuracy_dict[gra] = aux

        print(accuracy_dict)
        
        with open('kernel/accuracy_dict_iteration_' + 
                    str(it) + '.pkl', 'wb') as fp:
            pickle.dump(accuracy_dict, fp)

if __name__ == '__main__':
    main() 
