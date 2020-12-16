import data_manipulation as dm
from sklearn import datasets
from sklearn import model_selection


data = datasets.load_iris()['data']

offline_data, streaming_data = model_selection.train_test_split(data, test_size=0.6)

min_g = 1
max_g = 30
n_background = int(streaming_data.shape[0]*0.9)
Iteration = 1
laplace = 0

for gra in range(min_g, max_g):
    dm.SODA_Granularity_Iteration(offline_data,streaming_data,gra,n_background,Iteration,laplace)