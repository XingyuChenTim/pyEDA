from pyEDA.main import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Read file using pandas library
df = pd.read_csv('s4_EDA.csv', sep='  ', engine='python')
df = df.to_numpy()

#Extract baseline and stress data
base_st=int((5*60+52)*4)
base_end=int((25*60+3)*4)
stress_st=int((61*60+2)*4)
stress_end=int((72*60+15)*4)
df_1 = df[base_st:base_end]
df_2 = df[stress_st:stress_end]

#Define X (dataset without target column) and y (only target values)
time_test = np.arange(0, 32000, 0.25)
print (time_test.size)
X = np.append(df_1, df_2)
y = np.append(np.zeros(base_end-base_st), np.ones(stress_end-stress_st))
X=X.reshape(-1,1)

#Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 10)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=10)
#fit model to data
knn_gscv.fit(X, y)
#check top performing n_neighbors value
k = knn_gscv.best_params_
#check mean score for the top performing value of n_neighbors
score=knn_gscv.best_score_
#Print the optimal k value and accuracy
print (k, score)
# Create KNN classifier
for tem in k.values():
    k = tem
knn = KNeighborsClassifier(n_neighbors = k)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#############################################
#Use our model to predict unlabeled data from sample #
#############################################

#Read file using pandas library
df_test = pd.read_csv('s4_EDA.csv', sep='  ', engine='python')
eda = df_test.to_numpy()

#StatFeatureExtract
m, wd, eda_clean = process_statistical(eda, use_scipy=True, sample_rate=250, new_sample_rate=5, segment_width=10, segment_overlap=0)
eda_clean = np.array(eda_clean)
flat_eda_clean = [item for sublist in eda_clean for item in sublist]
flat_eda_clean = np.array(flat_eda_clean)

#Visualise the raw data
f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
time_series = []
for i in range (0, eda.size):
    time_series.append(i*0.25)
ax1.plot(time_series, eda, 'g', label = 'RAW GSR DATA')
ax1.legend()
ax1.grid()
ax1.set_xlabel('TIME (SECONDS)')
ax1.set_ylabel('SKIN CONDUCTANCE (MICROSIMENS)')
ax1.set_title('Raw data')

#Visualise the processed data
time_series = []
for i in range (0, flat_eda_clean.size):
    time_series.append(i*0.25*(eda.size/flat_eda_clean.size))
ax2.plot(time_series, flat_eda_clean, 'g', label = 'PREPROCESSED GSR DATA')
ax2.legend()
ax2.grid()
ax2.set_xlabel('TIME (SECONDS)')
ax2.set_ylabel('SKIN CONDUCTANCE (MICROSIMENS)')
ax2.set_title('Preprocessed data')
plt.show()

#use knn machine to predict unlabeled data
flat_eda_clean=flat_eda_clean.reshape(-1,1)
knn_pred_result = knn.predict(flat_eda_clean)

#Visualise the prediction
time_series_1 = []
for i in range (0, knn_pred_result.size):
    time_series_1.append(i*0.25*(eda.size/flat_eda_clean.size))
plt.plot(time_series_1, knn_pred_result, 'g', label = 'PREDICTED GSR DATA')
plt.legend()
plt.grid()
plt.xlabel('TIME (SECONDS)')
plt.ylabel('STRESS (0 OR 1)')
plt.title('Prediction')
plt.show()

#for keys,values in m.items():
#    print(keys)
#   print(values)
