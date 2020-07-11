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
from sklearn.cluster import KMeans
import statistics
from shutil import copy


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
        statistics_DF.loc[ind*3+layer_num] = [layer_num,
                                           round(statistics.mean(layers[layer_num]),3),
                                           round(statistics.mode(layers[layer_num]),3),
                                           round(statistics.median(layers[layer_num]),3),
                                           round(np.std(layers[layer_num]),3),
                                           round(np.max(layers[layer_num]),3),
                                           round(np.min(layers[layer_num]),3),
                                           num_of_neg, num_of_pos]


filename = ("D:\Projects\Python\100\BPopt_Nadam1\TableNadam19.csv")
names = ['n1', 'n2', '#',  'epoches', 'CDLMax', 'CDTMax', 'loss', 'r_square', 
         'val_loss', 'val_r_square', 'Top3Epoch']
dataset = pd.read_csv(r'D:\Projects\Python\100\BPopt_Nadam1\TableNadam19.csv', 
                      sep = ';' , names=names)  
# Формирование массива только с "хорошими" моделями (CDTMax>=0.5)

new_dataset = pd.DataFrame(columns=names)

indexes = []
ind = 0


for i in range(1,100):
    tmp = dataset['CDTMax'][i]
    if float(tmp) >= 0.5:
        new_dataset.loc[ind] = dataset.loc[i]
        ind+=1
        indexes.append(i)
        
print(indexes)  
dataset = new_dataset.copy()
    
    
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
ind=0

for num_of_model in indexes:
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
    layer_weights0 = weights0[0].flatten()
    #layer_weights0 = layer_weights0.flatten()
    layer_weights2 = weights2[0].flatten()
    layer_weights3 = weights3[0].flatten()
    
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
    all_weights_stat.loc[ind] = [round(statistics.mean(all_weights),3),
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
    ind+=1
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
    
    
    
  

mean_intervals_table = pd.DataFrame(columns=['mean', 'left_border', 'right_border'])

the_values = ['mean', 'mode', 'median','standart_deviation', 'max', 'min', 
                                         'num_of_negative', 'num_of_positive']


directories = ['all_weights', 'layer_weights0', 'layer_weights1', 
               'layer_weights2', 'bias_weights0', 'bias_weights1', 
               'bias_weights2']
num_of_clasters=[3, 4, 3, 2, 2, 2, 2]
L_colors = ['royalblue', 'midnightblue', 'blue', 'slateblue']
T_colors = ['lightcoral', 'brown', 'darkred', 'red']


#weights_statistics = [all_weights_stat, weight_layers[0], weight_layers[1], 
#                       weight_layers[2], bias_layers[0], bias_layers[1],
#                       bias_layers[2]]
for i in range(7):
    ind = 0
    w_stat = pd.DataFrame()
    if(i == 0):
        w_stat = all_weights_stat.copy()
    if(i == 1):
        w_stat = layer_weights_stat[0::3].copy()
    if(i == 2):
        w_stat = layer_weights_stat[1::3].copy()
    if(i == 3):
        w_stat = layer_weights_stat[2::3].copy()
    if(i == 4):
        w_stat = bias_weights_stat[0::3].copy()
    if(i == 5):
        w_stat = bias_weights_stat[1::3].copy()
    if(i == 6):
        w_stat = bias_weights_stat[2::3].copy()
    # кластеризация
    if i>1:
        w_stat.drop(columns=['num_of_layer'])
    np_w_stat = w_stat.to_numpy()
    kmeans = KMeans(n_clusters=num_of_clasters[i], init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(np_w_stat)
    for value in the_values:
        y=w_stat[value].to_numpy()
        # формирование массивов, учитывая распределение по кластерам
        x_axisCDL = []
        x_axisCDT = []
        x_axis_lossL = []
        x_axis_lossT = []
        y_axis = []
        models_in_claster = []
        for j in range(num_of_clasters[i]):
            x_axisCDL.append([])
            x_axisCDT.append([])
            x_axis_lossL.append([])
            x_axis_lossT.append([])
            y_axis.append([])
            models_in_claster.append([])
        y_pred = kmeans.predict(np_w_stat)
        for j in range(len(indexes)):
            x_axisCDL[y_pred[j]].append(dataset['CDLMax'][j])
            x_axisCDT[y_pred[j]].append(dataset['CDTMax'][j])
            x_axis_lossL[y_pred[j]].append(dataset['loss'][j])
            x_axis_lossT[y_pred[j]].append(dataset['val_loss'][j])
            y_axis[y_pred[j]].append(y[j])
            models_in_claster[y_pred[j]].append(indexes[j])
        #копирование картинок для моделей, учитывая распределение по кластерам
        for j in range(num_of_clasters[i]):
            if not os.path.exists(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\clasters\\'+str(j)):
                os.mkdir(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\clasters\\'+str(j))
            for k in models_in_claster[j]:
                copy(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\BP_9_5_10000_'+
                     str(k)+'.png', r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\clasters\\'+str(j))
                copy(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\logBP_9_5_'+
                     str(k)+'.png', r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\clasters\\'+str(j))
                copy(r'D:\Projects\Python\100\BPopt_Nadam1\№9_5\logBP_New_9_5_'+
                     str(k)+'.png', r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\clasters\\'+str(j))
        # рисование зависимостей стат. характеристик от метрик:
        fig = go.Figure()
        for j in range(num_of_clasters[i]):
            fig.add_trace(go.Scatter(
                   x=x_axisCDL[j],
                   y=y_axis[j],
                   marker=dict(color=L_colors[j], size=5),
                   mode="markers",
                   name="Обучающая (train), кластер "+str(j),
                   ))
            fig.add_trace(go.Scatter(
                   x=x_axisCDT[j],
                   y=y_axis[j],
                   marker=dict(color=T_colors[j], size=5),
                   mode="markers",
                   name="Тестовая (test), кластер "+str(j),
                   ))
        
        fig.update_layout(legend_orientation="h",legend=dict(x=0.1, y=-0.2), 
                          xaxis=dict(title=value,position=0.015),  
                          yaxis_title="Коэффициенты детерминации")
        fig.write_image(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\cd_'+value+'.png')
        
        fig = go.Figure()
        for j in range(num_of_clasters[i]):
            fig.add_trace(go.Scatter(
                   x=x_axis_lossL[j],
                   y=y_axis[j],
                   marker=dict(color=L_colors[j], size=5),
                   mode="markers",
                   name="Обучающая (train), кластер "+str(j),
                   ))
            fig.add_trace(go.Scatter(
                   x=x_axis_lossT[j],
                   y=y_axis[j],
                   marker=dict(color=T_colors[j], size=5),
                   mode="markers",
                   name="Тестовая (test), кластер "+str(j),
                   ))
        fig.update_layout(legend_orientation="h",legend=dict(x=0.1, y=-0.2), 
                          xaxis=dict(title=value,position=0.015),
                          yaxis_title="Ошибка")
        fig.write_image(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\loss_'+value+'.png')
        
        # рисование гистограмм для стат. метрик
        figure, ax = plt.subplots()
        ax.set_title('Гистограмма для характеристики '+value)
        ax.hist(y, bins=20)
        plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\histogramm_'+value+'.png')
        plt.close(figure)
        plt.clf()
        
        # расчет доверительных интервалов для стат. метрик
        mean = y.mean()
        left, right = st.t.interval(0.95, len(y)-1, 
                                    loc=np.mean(y), 
                                    scale=st.sem(y))
        mean_intervals_table.loc[ind] = [mean, left, right]
        ind+=1
        mean_intervals_table.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                                    directories[i]+'\intervals.csv',
                     index=False, sep=';')
    # кластеризация
    """
    wcss = [] #Within Clusters Sum of Squares
    print(w_stat)
    if i>1:
        w_stat.drop(columns=['num_of_layer'])
    w_stat = w_stat.to_numpy()
    print(w_stat)
    #поиск оптимального числа кластеров
    for n_cl in range(1, 11):
        kmeans = KMeans(n_clusters=n_cl, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(w_stat)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(r'D:\Projects\Python\100\BPopt_Nadam1\\'+
                directories[i]+'_KMeans_check.png')
    """
    
        
all_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\all_weights.csv',
                 index=False, sep=';')
layer_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\layer_weights.csv',
                 index=False, sep=';') 
bias_weights_stat.to_csv(r'D:\Projects\Python\100\BPopt_Nadam1\bias_weights.csv',
                 index=False, sep=';') 







