from pyEDA.main import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

#Read file using pandas library
df = pd.read_csv('s4_EDA.csv', sep='  ', engine='python')
#Use sample size = 15000
df = df[4:32000]
#Convert numpy array into one array
eda = df.values.flatten()
#Convert string value into float value
eda = eda.astype(np.float)
#Divide all values by 1000 to get correct data
#eda_signal = np.divide(eda, 1000)
#Print the sample data in a list
#print (eda_signal)
print (eda)

# Visualise the data
plt.plot(eda)
plt.show()

#StatFeatureExtract
m, wd, eda_clean = process_statistical(eda_sign, use_scipy=True, sample_rate=250, new_sample_rate=40, segment_width=10, segment_overlap=0)

for keys,values in m.items():
    print(keys)
    print(values)
