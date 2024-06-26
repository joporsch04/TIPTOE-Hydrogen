import csv
import matplotlib.pyplot as plt
from scipy import interpolate as intp
import numpy as np
import pandas as pd

x = []
y = []

dicField={}
time_array=np.arange(-1000, 1000+0.25, 0.25)
dicField['t_au']=time_array
            
def interPol(dataframe):
    x = dataframe.iloc[:, 0]
    y = dataframe.iloc[:, 1:]

    for column in y.columns:
        y_val = y[column]
        fct = intp.interp1d(x, y_val)
        dicField[column] = fct(x)
    
    return dicField, x


data, x = interPol(pd.read_csv('output_extract_field_0.csv'))

safe_data = pd.DataFrame(data)
safe_data.to_csv(data, 'output_interpol.csv')

#intfct = interp1d(np.array(x), np.array(y))

#dicField[f'Field_{del}']=intfct(time_array)


#coeff = np.polyfit(np.array(x), np.array(y), len(x)-1)


#x_new = np.linspace(1, len(x), 2*len(x))
#y_new = intfct(x_new)

#plt.plot(x, y)
#plt.plot(x_new, y_new)
#print(coeff)


plt.show()