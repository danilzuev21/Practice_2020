import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
#from tensorflow.keras.layers import advanced_activations
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import optimizers
from keras import regularizers
import plotly
import plotly.graph_objects as go
import os
from tensorflow.keras import initializers as initi
from tensorflow.keras import backend as K
from scipy import stats as st
# from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
# Для избежания  проблем с открытием файлов:
from pathlib import Path



# Метрика R-квадрат
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/SS_tot)

n1 = 9
n2 = 5
gl1=0.65*np.random.randn(10,n1)
gl11=np.random.random_sample((n1,))-1.5
gl2= 0.65*np.random.randn(n1,n2)
gl22=np.random.random_sample((n2,))-1.5
gl3=0.65*np.random.randn(n2,1)  
gl33=np.array([0.35])

model = models.Sequential()
model.add(layers.Dense(n1, input_dim=10, activation = 'tanh', kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None), use_bias=True,  bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None)))
model.add(layers.Dense(n2, activation = 'tanh', kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None), use_bias=True,  bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None)))
               # Output- Layer
model.add(layers.Dense(1,activation = 'elu', kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None), use_bias=True,  bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None)))
model.layers[0].set_weights([gl1,gl11])
model.layers[1].set_weights([gl2,gl22])
model.layers[2].set_weights([gl3,gl33])

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print('next model:')

# Проход по всем моделям
for num_of_model in range(1):
    initialized_model = r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\initBP_9_5_'+str(num_of_model)+'.h5'
    trained_model = r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\BP_9_5_'+str(num_of_model)+'.h5'
    init_model = load_model(initialized_model, custom_objects={'r_square':r_square})
    model = load_model(trained_model, custom_objects={'r_square':r_square})
    init_weights0 = init_model.layers[0].get_weights()
    init_weights1 = init_model.layers[1].get_weights()# пустой массив, далее рассматриваться не будет
    init_weights2 = init_model.layers[2].get_weights()
    weights0 = model.layers[0].get_weights()
    weights1 = model.layers[1].get_weights()# пустой массив, далее рассматриваться не будет
    weights2 = model.layers[2].get_weights()
    weights3 = model.layers[3].get_weights()
    #print(weights0)
    #print(weights1)
    #print(weights2)
    print(weights3)
    