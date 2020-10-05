# Anomaly_Detection
## SODA applied to anomaly detection 
The SODA is a self-organized algorithm which, partitions a data-set into non-parametric data-clouds. In the offline mode, we deliver to SODA a data-set compound only by normal events (background). Afterwards, in its online mode, SODA re-organizes those data-clouds to follow the streaming data patterns. Thus, by analyzing the difference between data clouds before and after a streaming data arrival, one can identify anomaly data patterns (from the signal). This analysis calculates how much of the offline data is inside each data-cloud after the streaming data arrival. Consequently, data-cloud with more offline data is more similar to normal events (background). Those with no offline data are regarded as anomaly data-clouds since they don't follow the offline data patterns.

### Content

* Anomaly_Detection.py: The most recent version of our model;
* SODA.py: A python version of SODA routines and processes;
* data_manipulation.py: A library containing all the routines and processes employed by our model;  
* results: Folder containing the outputs of `Anomaly_Detection.py` script;
* Analysed_Signal: Folder containing the data and labels of the detected anomalies;

### python dependencies

Our model depends on the following python libraries :

* numpy
* import time
* pandas
* import pickle
* math
* sklearn
* matplotlib
* scipy
* os
* rogress
* mport threading
* atetime 

### To do list

- [ ] Run our script for larger data-sets;
- [ ] Employ hypothesis test;
