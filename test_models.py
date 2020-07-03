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
import statistics



# Метрика R-квадрат
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/SS_tot)


def weight_statistics(layers, statistics_DF):
    for layer_num in range(3):
        num_of_neg = 0
        num_of_pos = 0
        for number in layers[layer_num]:
            if number > 0:
                num_of_neg+=1
            else:
                num_of_pos+=1
        statistics_DF.loc[num_of_model*3+layer_num] = [layer_num,
                                           round(statistics.mean(layers[layer_num]),3),
                                           round(statistics.mode(layers[layer_num]),3),
                                           round(statistics.median(layers[layer_num]),3),
                                           round(np.std(layers[layer_num]),3),
                                           round(np.max(layers[layer_num]),3),
                                           round(np.min(layers[layer_num]),3),
                                           num_of_neg, num_of_pos]


all_weights_stat = pd.DataFrame(columns=['mean', 'mode', 'median', 
                                         'standart_deviation', 'max', 'min', 
                                         'num_of_negative', 'num_of_positive'])
layer_weights_stat = pd.DataFrame(columns=['num_of_layer','mean', 'mode', 
                                           'median', 'standart_deviation', 
                                           'max', 'min', 'num_of_negative', 
                                           'num_of_positive'])
bias_weights_stat = pd.DataFrame(columns=['num_of_layer', 'mean', 'mode', 
                                          'median', 'standart_deviation', 'max', 
                                          'min', 'num_of_negative', 
                                          'num_of_positive'])


# Проход по всем моделям
for num_of_model in range(100):
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
    # веса слоев
    layer_weights0 = np.concatenate([weights0[0].flatten(), weights0[1].flatten()])
    #layer_weights0 = layer_weights0.flatten()
    layer_weights2 = np.concatenate([weights2[0].flatten(), weights2[1].flatten()])
    layer_weights3 = np.concatenate([weights3[0].flatten(), weights3[1].flatten()])
    
    bias_wights0 = weights0[1]
    bias_wights2 = weights2[1]
    bias_wights3 = weights3[1]
    all_weights = np.concatenate([layer_weights0, layer_weights2, layer_weights3])
    num_of_neg = 0
    num_of_pos = 0
    for number in all_weights:
        if number > 0:
            num_of_neg+=1
        else:
            num_of_pos+=1
    all_weights_stat.loc[num_of_model] = [round(statistics.mean(all_weights),3),
                                           round(statistics.mode(all_weights),3),
                                           round(statistics.median(all_weights),3),
                                           round(np.std(all_weights),3),
                                           round(np.max(all_weights),3),
                                           round(np.min(all_weights),3),
                                           num_of_neg, num_of_pos]
    weight_layers=[layer_weights0, layer_weights2, layer_weights3]
    weight_statistics(weight_layers, layer_weights_stat)
    bias_layers=[bias_wights0, bias_wights2, bias_wights3]
    weight_statistics(bias_layers, bias_weights_stat)
    # гистограммы по весам слоев:
    fig, ax = plt.subplots()
    ax.set_title('Гистограмма для весов всех слоев')
    bins=17
    ax.hist(all_weights, bins=bins)
    plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\histograms\all_weights_hist_'+str(num_of_model)+'.png')
    plt.close(fig)
    plt.clf()
    for i in range(3):
        fig, ax = plt.subplots()
        ax.set_title('Гистограмма для весов слоя '+str(i))
        ax.hist(weight_layers[i], bins=bins)
        plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\histograms\layer_weights_hist_'+str(num_of_model)+'_'+str(i)+'.png')
        plt.close(fig)
        plt.clf()
        
        fig, ax = plt.subplots()
        ax.set_title('Гистограмма для весов смещения слоя '+str(i))
        ax.hist(bias_layers[i], bins=bins)
        plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\histograms\bias_layer_weights_hist_'+str(num_of_model)+'_'+str(i)+'.png')
        plt.close(fig)
        plt.clf()
    
    
    
filename = ("D:\Projects\Python\100\BPopt_Nadam1\TableNadam19.csv")
names = ['n1', 'n2', '#',  'epoches', 'CDLMax', 'CDTMax', 'loss', 'r_square', 'val_loss', 'val_r_square', 'Top3Epoch']
dataset = pd.read_csv(r'D:\Projects\Python\100\BPopt_Nadam1\TableNadam19.csv', sep = ';' , names=names)    

mean_intervals_table = pd.DataFrame(columns=['mean', 'left_border', 'right_border'])

the_values = ['mean', 'mode', 'median','standart_deviation', 'max', 'min', 
                                         'num_of_negative', 'num_of_positive']
ind = 0
for value in the_values:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
               x=dataset['CDLMax'],
               y=all_weights_stat[value],
               marker=dict(color="blue", size=5),
               mode="markers",
               name="Обучающая (train)",
               ))
    fig.add_trace(go.Scatter(
               x=dataset['CDTMax'],
               y=all_weights_stat[value],
               marker=dict(color="red", size=5),
               mode="markers",
               name="Тестовая (test)",
               ))
    fig.update_layout(legend_orientation="h",legend=dict(x=0.1, y=-0.2), xaxis=dict(title=value,position=0.015),  yaxis_title="Коэффициенты детерминации")
    fig.write_image(r'D:\Projects\Python\100\BPopt_Nadam1\cd_'+value+'.png')
    
    figure, ax = plt.subplots()
    ax.set_title('Гистограмма для характеристики '+value)
    ax.hist(all_weights_stat[value], bins=20)
    plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\histograms\histogramm_'+value+'.png')
    plt.close(figure)
    plt.clf()
    
    mean = all_weights_stat[value].mean()
    left, right = st.t.interval(0.95, len(all_weights_stat[value])-1, 
                                loc=np.mean(all_weights_stat[value]), 
                                scale=st.sem(all_weights_stat[value]))
    mean_intervals_table.loc[ind] = [mean, left, right]
    ind+=1
    mean_intervals_table.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\intervals.csv',
                 index=False, sep=';')
        
all_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\all_weights.csv',
                 index=False, sep=';')
layer_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\layer_weights.csv',
                 index=False, sep=';') 
bias_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\bias_weights.csv',
                 index=False, sep=';') 
    #print(weights0)
    #print(weights1)
    #print(weights2)
    #print(model.summary())
    