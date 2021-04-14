from pyEDA.main import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read file using pandas library
df = pd.read_csv('sample.csv', sep='  ', engine='python')
#Use sample size = 15000
df = df[1:15001]
#Convert numpy array into one array
eda = df.values.flatten()
#Convert string value into float value
eda = eda.astype(np.float)
#Divide all values by 1000 to get correct data
eda_signal = np.divide(eda, 1000)
#Split array into multiple (10) array 
eda_signals = np.split(eda_signal, 10)

# Visualise the list of signals
print(eda_signals)
print(np.array(eda_signals).shape)

# Visualise one of the signals in the list
plt.plot(eda_signals[0])
plt.show()

prepare_automatic(eda_signals, sample_rate=250, new_sample_rate=250, k=32, epochs=100, batch_size=10)

automatic_features = process_automatic(eda_signals[0])

print(automatic_features)
