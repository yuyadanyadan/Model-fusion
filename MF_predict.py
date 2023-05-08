
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import time
import datetime

from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, KFold

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout
from scipy import stats

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def R2_error(y_true, y_pred):
    #y_true_avg = np.mean(y_true)
    score = r2_score(y_true, y_pred, multioutput= 'uniform_average')
    return score

def computeError(y_true, y_pred, isprint):
    error_mape = mean_absolute_percentage_error(y_true, y_pred)
    error_rmse = rmse(y_true, y_pred)
    error_R2 = R2_error(y_true, y_pred)
    if isprint:
        print("mape:", "%.2f"%error_mape, " rmse: ",  error_rmse , " R2: ", error_R2)
    return error_R2

#将天气数据中为空的数据，利用前后为非空的数据的均值来进行补充；
def preprocess_nandata(data_list, max_value):
    new_data_list = data_list.copy()
    commit_num = 0
    for i in range(len(data_list)):
        data = data_list[i]
        if math.isnan(data) or data > max_value:
            j = i-1
            while math.isnan(data_list[j]):
                j = j-1
            before_data = data_list[j]
            j1 = i+1
            while j1<data_list.shape[0] and math.isnan(data_list[j1]):
                j1 = j1+1
            if j1>data_list.shape[0]-1:
                j1 = j
            after_data = data_list[j1]
            new_data_list[i] = (before_data+after_data)/2
            commit_num = commit_num + 1

    return new_data_list  

def load_weather_data(filename):
    data = pd.read_excel(filename, usecols=[0,1,2,3])
    data_T = data['T'].values
    data_Rain = data['Rain'].values
    data_Time = data['time'].values
    
    #对数据中NaN数据进行处理   

    maxT = 100    
    data_T = preprocess_nandata(data_T, maxT)

    maxRain = 1000
    data_Rain = preprocess_nandata(data_Rain, maxRain)
    
    data_T = data_T.reshape(-1,1)
    data_Rain = data_Rain.reshape(-1,1)

    weather_feas = np.append(data_T, data_Rain, axis=1)
    weather_feas = weather_feas.astype(float)
    
    # plt.plot( data_Time, data_T, 'ro-')
    # plt.xlabel('Time')
    # plt.ylabel('Temperature')
    # #plt.title('Temperature')
    # #plt.legend()
    # plt.show()
    
    # plt.plot( data_Time, data_Rain, 'ro-')
    # plt.xlabel('Time')
    # plt.ylabel('Rainfall')
    # #plt.title('Temperature')
    # #plt.legend()
    # plt.show()
    
    return weather_feas, data_Time

def is_valid_date(date, data_format):
    try:
        time.strptime(date.strftime(data_format), data_format)
        return True
    except Exception:
        return False

def preprocess_outlier(cod_list, min_value, max_value):
    new_cod_list = cod_list.copy()
    commit_num = 0
    for i in range(len(cod_list)):
        cod = cod_list[i]
        if cod < min_value or cod > max_value:
            j = i-1
            while cod_list[j] < min_value or cod_list[j] > max_value:
                j = j-1
            before_cod = cod_list[j]
            j1 = i+1
            while j1<cod_list.shape[0] and cod_list[j1] < min_value or cod_list[j1] > max_value:
                j1 = j1+1
            if j1>cod_list.shape[0]-1:
                j1 = j
            after_cod = cod_list[j1]
            new_cod_list[i] = (before_cod+after_cod)/2
            commit_num = commit_num + 1
            #print("commit cod before: ", cod, " after: ", new_cod_list[i])

    return new_cod_list     

#对进水数据进行处理，当进水数据为0时，将前后最近的两个不为0的数据的均值作为当前的值
#输入数据是list类型，列向量
def preprocess_zerodata(cod_list):
    new_cod_list = cod_list.copy()
    commit_num = 0
    for i in range(len(cod_list)):
        cod = cod_list[i]
        if cod == 0.0:
            j = i-1
            while cod_list[j] == 0.0:
                j = j-1
            before_cod = cod_list[j]
            j1 = i+1
            while j1<cod_list.shape[0] and cod_list[j1] == 0.0:
                j1 = j1+1
            if j1>cod_list.shape[0]-1:
                j1 = j
            after_cod = cod_list[j1]
            new_cod_list[i] = (before_cod+after_cod)/2
            commit_num = commit_num + 1
            #print("commit cod before: ", cod, " after: ", new_cod_list[i])
    print("zero commit_num: ", commit_num)

    return new_cod_list     

#从表格中获得名字为value_str的列的值
def getvalues(data1, value_str):
    values1 = data1[value_str]
    values = []
    for i in range(len(values1)):
        #判断当前的数据类型是否是字符串类型
        if type(values1[0])==type("abc"):
            if values1[i][0]==str('#'):
                values1[i] = "0.0"
        values.append(float(values1[i]))
        
    return values

def load_sewage_data(sewage_file_list, isTuodong):
    print("sewage_file_list: ", sewage_file_list)
    list_file = open(sewage_file_list)
    print(list_file)
    list_file.seek(0)
    list_file.read()
    list_file.seek(0)

    feas = []    
    date = []
    strs = []
    if isTuodong:
       #瞬时流量、进水PH、进水氨氮、进水总磷、进水COD、进水总氮；
       strs = ['瞬时流量', '进水PH', '进水氨氮', '进水总磷', '进水COD', '进水总氮']
       feas = np.empty((0, 6))
    else:
       #瞬时流量、进水PH、进水氨氮、进水总磷、进水COD；
       strs = ['瞬时流量', '进水PH', '进水氨氮', '进水总磷', '进水COD']
       feas = np.empty((0, 5))
    
    for excel_name in list_file:
        pathname =  excel_name
        print("pathname: ", pathname)
        dat = pd.read_excel(pathname.strip('\n'), sheet_name=0)
        date_format = "%Y-%m-%d %H:%M:%S"
        for row in dat.values:
            if row[0] == row[0]:
                if is_valid_date(row[0], date_format):
                   #print("row[0]: ", row[0], " type: ", )
                   date.append(row[0])
        
        data1 = pd.read_excel(pathname.strip('\n'), sheet_name=0)
        sub_feas = np.empty((0, 0))
        for value_str in strs:
           sub_values = getvalues(data1,value_str)
           if sub_feas.shape[0] == 0:
              sub_feas = np.zeros((len(sub_values),0))
           sub_feas = np.insert(sub_feas, sub_feas.shape[1], values=sub_values, axis=1)
        
        feas = np.insert(feas, feas.shape[0], values=sub_feas, axis=0)

    list_file.close()
    
    #对于特征为0的数据进行处理,碰到特征为0的数据，将前后特征不为0的数据求平均;
    new_feas = np.zeros((feas.shape[0], 0))
    for i in range(feas.shape[1]):
       feas_col = feas[:, i]
       new_feas_col = preprocess_zerodata(feas_col)
       
       new_feas = np.append(new_feas, new_feas_col.reshape(-1,1), axis=1)

    min_value = 0
    max_value = 11
    new_feas[:,1] = preprocess_outlier(new_feas[:,1], min_value, max_value)    

    return new_feas, date

def align_weather_sewage(weather_feas, weather_time, sewage_feas, sewage_time, datacols):
   print("weather_feas.shape: ", weather_feas.shape, " weather_time.shape: ", len(weather_time), 
   " sewage_feas.shape: ", sewage_feas.shape, " sewage_time.shape: ", len(sewage_time))
   j = 0
   align_time = []
   #输入特征：温度、降水、温度*降水，温度*温度，降水*降水
   feas_num = 2
   x = np.zeros((0,feas_num))
   #输出变量：瞬时流量、进水PH、进水氨氮、进水总磷、进水COD、进水总氮
   y = np.zeros((0,datacols))
   
   for i in range(len(weather_time)):
      #时间字符串转时间
      weather_t = weather_time[i]
      #当j超过范围时，直接返回;
      if j > len(sewage_time)-1:
         break
      sewage_t = sewage_time[j]
      
      if weather_t < sewage_t:
         continue
      elif weather_t > sewage_t:
         while j < len(sewage_time) and weather_time[i] != sewage_time[j]:
            j = j + 1
      
      if j < len(sewage_time):
         sewage_t = sewage_time[j]
      
         #此时的weather_t和sewage_t应该是相等的
         if weather_t == sewage_t:      
            align_time.append(weather_time[i])
            x1 = weather_feas[i,0]
            x2 = weather_feas[i,1]
            fea = [x1, x2]
            x = np.append(x, (np.array([fea])).reshape(1,feas_num), axis=0)
            y = np.append(y, (sewage_feas[j,:]).reshape(1,datacols), axis=0)

   print("after align x.shape: ", x.shape, " y.shape: ", y.shape)
   return align_time, x, y  

def save_predict_data2( y_true, y_pred, filename):
    #计算误差
    rows = y_pred.shape[0]
    df_y_true = pd.DataFrame({'y_true':y_true.reshape(rows)})
    df_y_pred = pd.DataFrame({'y_pred':y_pred.reshape(rows)})
    

    writer = pd.ExcelWriter(filename)
    df_y_true.to_excel(writer, sheet_name='Predict', startcol=0, index=False)
    df_y_pred.to_excel(writer, sheet_name='Predict', startcol=1, index=False)

    writer.save()

def save_predict_data( pred_y, error_ratios, filename):
    #计算误差
    y_true = pred_y[0,:]
    y_svr = pred_y[1,:]   
    y_knn = pred_y[2,:]
    y_decisionTree = pred_y[3,:]
    y_randomForest = pred_y[4,:]
    y_gbdt = pred_y[5,:]
    y_linear = pred_y[6,:]
    y_ridge = pred_y[7,:]
    y_lasso = pred_y[8,:]
    y_elastic = pred_y[9,:]
    y_keras = pred_y[10,:]
    print("svr:")
    computeError(y_true, y_svr, 1)
    print()
    print("knn:")
    computeError(y_true, y_knn, 1)
    print()
    print("decisionTree:")
    computeError(y_true, y_decisionTree, 1)
    print()
    print("randomForest:")
    computeError(y_true, y_randomForest, 1)
    print()
    print("gbdt:")
    computeError(y_true, y_gbdt, 1)
    print()
    print("linear:")
    computeError(y_true, y_linear, 1)
    print()
    print("ridge:")
    computeError(y_true, y_ridge, 1)
    print()
    print("lasso:")
    computeError(y_true, y_lasso, 1)
    print()
    print("elastic:")
    computeError(y_true, y_elastic, 1)
    print()
    print("keras:")
    computeError(y_true, y_keras, 1)
    
    rows = pred_y.shape[1]
    #df_date = pd.DataFrame({'Date':date})
    df_y_test = pd.DataFrame({'y_test':pred_y[0,:].reshape(rows)})
    df_y_svr = pd.DataFrame({'y_svr':pred_y[1,:].reshape(rows)})
    df_y_knn = pd.DataFrame({'y_knn':pred_y[2,:].reshape(rows)})
    df_y_decisionTree = pd.DataFrame({'y_decisionTree':pred_y[3,:].reshape(rows)})
    df_y_randomForest = pd.DataFrame({'y_randomForest':pred_y[4,:].reshape(rows)})
    df_y_gbdt = pd.DataFrame({'y_gbdt':pred_y[5,:].reshape(rows)})
    df_y_linear = pd.DataFrame({'y_linear':pred_y[6,:].reshape(rows)})
    df_y_ridge = pd.DataFrame({'y_ridge':pred_y[7,:].reshape(rows)})
    df_y_lasso = pd.DataFrame({'y_lasso':pred_y[8,:].reshape(rows)})
    df_y_elastic = pd.DataFrame({'y_elastic':pred_y[9,:].reshape(rows)})
    df_y_keras = pd.DataFrame({'y_keras':pred_y[10,:].reshape(rows)})

    writer = pd.ExcelWriter(filename)
    #df_date.to_excel(writer, sheet_name='Predict', startcol=0, index=False)
    df_y_test.to_excel(writer, sheet_name='Predict', startcol=0, index=False)
    df_y_svr.to_excel(writer, sheet_name='Predict', startcol=1, index=False)
    df_y_knn.to_excel(writer, sheet_name='Predict', startcol=2, index=False)
    df_y_decisionTree.to_excel(writer, sheet_name='Predict', startcol=3, index=False)
    df_y_randomForest.to_excel(writer, sheet_name='Predict', startcol=4, index=False)
    df_y_gbdt.to_excel(writer, sheet_name='Predict', startcol=5, index=False)
    df_y_linear.to_excel(writer, sheet_name='Predict', startcol=6, index=False)
    df_y_ridge.to_excel(writer, sheet_name='Predict', startcol=7, index=False)
    df_y_lasso.to_excel(writer, sheet_name='Predict', startcol=8, index=False)
    df_y_elastic.to_excel(writer, sheet_name='Predict', startcol=9, index=False)
    df_y_keras.to_excel(writer, sheet_name='Predict', startcol=10, index=False)
    
    writer.save()

class Kerasnn_model:
    def __init__(self, inputdim):
        self.model = Sequential()
        from keras.layers import Dense 
        self.model.add(Dense(units=8, activation='relu', input_dim=inputdim))
        self.model.add(Dense(units=16, activation='relu'))
        self.model.add(Dense(units=4, activation='relu'))
        self.model.add(Dense(units=1))
        #adam优化器的默认学习率为0.001
        adam = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='mean_squared_error',optimizer=adam)
    def fit(self, X, y):
        #verbose:默认为1，显示进度条，0表示不在标准输出流输出日志信息，2表示每个epoch输出一行记录
        history1 = self.model.fit(X, y, validation_data=(X, y),epochs=30, batch_size=16, verbose=0)
    def predict(self,X_test):
        y_predict=self.model.predict(X_test, batch_size=16)
        return y_predict
    
 #batch_size:表示一次训练样本的数目,
 #如果是小数据集，建议一次性把所有数据都训练完。如果是大数据集，不能使用全批次
 #一个恰当的batch_size值，会以batch_size大小的数据一次性输入网络中，然后计算这个batch所有样本的平均损失，即代价函数是所有样本的平均；
 #优点：通过并行化提高内存利用率,单词epoch的迭代次数减少，提高运行速度。
 #单次epoch=(全部训练样本/batch_size)/iteration=1
 #适当增加batch_size,梯度下降方向准确性增加，训练震动的幅度减少
def kerasnn(x_tran, y_train, x_val, y_val, x_test, y_test):
    model = Sequential()
    from keras.layers import Dense 
    #cod
    #model.add(Dense(units=256, activation='relu', input_dim=np.array(x_tran).shape[1]))
    #model.add(Dense(units=512, activation='relu'))
    #model.add(Dense(units=256, activation='relu'))
    #总磷
    #model.add(Dense(units=256, activation='relu', input_dim=np.array(x_tran).shape[1]))
    #model.add(Dense(units=512, activation='relu'))
    #model.add(Dense(units=256, activation='relu'))
    #model.add(Dense(units=64, activation='relu'))
    #氨氮
    model.add(Dense(units=8, activation='relu', input_dim=np.array(x_tran).shape[1]))
    model.add(Dense(units=16, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(units=512, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(units=256, activation='relu'))
    #model.add(Dropout(0.2))
   # model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=1))
    #adam优化器的默认学习率为0.001
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error',optimizer=adam)
    
    
    #verbose:默认为1，显示进度条，0表示不在标准输出流输出日志信息，2表示每个epoch输出一行记录
    history1 = model.fit(x_tran, y_train, validation_data=(x_val,y_val),epochs=30, batch_size=16, verbose=0)
        
    loss_and_metrics=model.evaluate(x_val, y_val, batch_size=16)
    accu1=model.evaluate(x_test, y_test, batch_size=16)
    accu0=model.evaluate(x_tran, y_train, batch_size=16)
    #print("train_accu: ", accu0, " val_accurate: ", loss_and_metrics, " test_accu: ", accu1)
    
    # print(history1.history.keys())
    lossy = history1.history['loss']
    val_lossy = history1.history['val_loss']
    # accuy = history1.history['accuracy']
    # val_accuy = history1.history['val_accuracy']
    # plt.subplot(211)
    plt.plot(lossy, color='r', label="train_loss")
    plt.plot(val_lossy, color='b', label="val_loss")
    #plt.subplot(212)
    #plt.plot(accuy, color='r', label="train_acc")
    #plt.plot(val_accuy, color='b', label="val_acc")
    plt.legend()
    plt.show()
   
    #保存模型
    model.save_weights("my_model.h5")
    
    #加载模型
    model.load_weights("my_model.h5")
    
    #预测
    y_predict=model.predict(x_test, batch_size=16)
    return y_predict
 

#预测时候做模型融合，对于COD来说，通过R2来看，选择SVR, DecisionTree ,RandomForest GBDT 和NN;
def predict_MF(x,y,outfile):
    #将数据利用standardScaler进行归一化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
                  
    #模型融合中使用到的各个单个模型 COD
    regressors = [SVR(kernel='rbf', gamma=0.01, C=1.25),
                  #KNeighborsRegressor(n_neighbors=5),
                  #linear_model.LinearRegression(),
                  #DecisionTreeRegressor(max_depth=5),
                  #ensemble.RandomForestRegressor(),
                  #ensemble.GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=10),
                  Kerasnn_model(x.shape[1])]
                  
    rkf0 = KFold(n_splits=10)
    x_orig = x
    y_orig = y
    X_0,X_test,y_0,y_test = train_test_split(x_orig, y_orig, test_size=0.1, random_state=0)
    bestR2score = 0.0
    for m, (train0, test0) in enumerate(rkf0.split(X_0,y_0)):
        begin_time = time.time()
        X,y, X_predict, y_predict = X_0,y_0,X_test, y_test;

        y_predict = scaler.inverse_transform(y_predict)
    
        #训练data_blend_train, 来预测dataset_blend_test
        dataset_blend_train = np.zeros((X.shape[0], len(regressors)))
        dataset_blend_test = np.zeros((X_predict.shape[0], len(regressors)))
        error_test = np.zeros((1,len(regressors)))
   
        n_split = 5+m
        rkf = KFold(n_splits=n_split)

    
        for j, clf in enumerate(regressors):
            #依次训练各个单模型
            dataset_blend_test_j = np.zeros((X_predict.shape[0],n_split))
            error_regressor = np.zeros((1,n_split))
            for i, (train, test) in enumerate(rkf.split(X,y)):
                #使用第i个部分作为预测，剩余的部分用来训练模型，获得其预测的输出作为第i部分的新特征
                X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            
                clf.fit(X_train, y_train)
                y_submission = clf.predict(X_test)
                #print("dataset_blend_train[test,j].shape: ", dataset_blend_train[test,j].shape, " y_submission.shape: ", y_submission.shape)
                dataset_blend_train[test,j] = y_submission.flatten()
                dataset_blend_test_j[:,i] = clf.predict(X_predict).flatten();

                # print("第i折：",i)
                inv_predict_y = scaler.inverse_transform((dataset_blend_test_j[:,i]).reshape(-1,1))
                fold_R2 = computeError(y_predict, inv_predict_y, 0)
                error_regressor[0,i] = fold_R2
            #对于测试集，直接将这k折个模型的预测均值作为新的特征
            error_regressor = error_regressor/np.sum(error_regressor)
            dataset_blend_test[:,j] = (np.dot(dataset_blend_test_j, np.transpose(error_regressor))).flatten()  
    
        #最后预测；
        clfs = [linear_model.LinearRegression(),
                  #SVR(kernel='rbf', gamma=0.01, C=1.25),
                  #KNeighborsRegressor(n_neighbors=5)
                  ]
        for t, clf in enumerate(clfs):
            clf.fit(dataset_blend_train,y)
            y_submission = clf.predict(dataset_blend_test)
            y_pred = scaler.inverse_transform(y_submission.reshape(-1,1))
            final_R2 = computeError(y_predict, y_pred, 1)
            if final_R2 > bestR2score:
               bestR2score = final_R2
               filename =outfile+str(n_split)+".xlsx"
               save_predict_data2( y_predict, y_pred, filename)
        end_time = time.time()
        used_time = (end_time - begin_time)
        print("耗时：{:.2f}秒".format(end_time - begin_time))
    

def predict(x,y, outfile):
    #将数据利用standardScaler进行归一化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
    keras_error_sum = 0.0
    knn_error_sum = 0.0
    svr_error_sum = 0.0
    linear_error_sum = 0.0
    decisionTree_error_sum = 0.0
    randomForest_error_sum = 0.0
    gbdt_error_sum = 0.0
    
    #从训练数据中分出一部分作为验证数据集
    x_train_all,x_test,y_train_all,y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    # print("x_train_all.shape: ", x_train_all.shape, " y_train_all.shape: ", y_train_all.shape)
    # print("x_test.shape: ", x_test.shape)
    # print("y_test.shape: ", y_test.shape)
    
    y_test = scaler.inverse_transform(y_test)
    #print("after y_test.shape: ", y_test.shape)
    
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=10)
    num = 0
    y_results = np.zeros((0,0))
    
    RF_grid = []
    
    for train_index, val_index in rkf.split(x_train_all):
        if num > 0:
            break;
        num = num + 1

        #print("-------num-------: ", num)
        x_train, y_train = x_train_all[train_index],y_train_all[train_index]
        x_val, y_val = x_train_all[val_index],y_train_all[val_index]

        #print("x_val.shape: ", x_val.shape, " x_train.shape: ", x_train.shape)
        
        #训练+预测
        #svr回归，核函数为sigmoid时，不管对于哪个池子的过程都不适用;
        #kernel函数有以下几种:'linear', 'poly', 'rbf', 'sigmoid'
        #clf = SVR(kernel='linear', C=1.25)
        #gamma可以选择0.01
        clf = SVR(kernel='rbf', gamma=0.01, C=1.25)
        clf.fit(x_train, y_train)
        y_svr = clf.predict(x_test)
        
        #最近邻回归
        knnrgr = KNeighborsRegressor(n_neighbors=5)
        knnrgr.fit(x_train,y_train)
        y_knn = knnrgr.predict(x_test)
        
        #线性模型
        linearmodel = linear_model.LinearRegression()
        linearmodel.fit(x_train, y_train)
        y_linear = linearmodel.predict(x_test)
        
        #决策树模型
        decisionTreemodel = DecisionTreeRegressor(max_depth=5)
        decisionTreemodel.fit(x_train, y_train)
        y_decisionTree = decisionTreemodel.predict(x_test)
        
        #随机森林
        #randomForest_regr = ensemble.RandomForestRegressor(n_estimators = 50, max_depth=50, random_state=0)
        randomForest_regr = ensemble.RandomForestRegressor()
        randomForest_regr.fit(x_train, y_train)
        y_randomForest = randomForest_regr.predict(x_test)
        
        #gbdt模型
        gbdt = ensemble.GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=10)
        gbdt.fit(x_train, y_train)
        y_gbdt = gbdt.predict(x_test)
        
        #通过keras库，来搭建神经网络，
        y_keras=kerasnn(x_train, y_train, x_val, y_val, x_test, y_test)
        

        #print("after y_test: ", y_test)
        y_svr = y_svr.reshape(-1,1)
        y_svr = scaler.inverse_transform(y_svr)
        svr_error_ratio = mean_absolute_percentage_error(y_test, y_svr)
        
        y_knn = scaler.inverse_transform(y_knn)
        y_knn = y_knn.reshape(-1,1)
        #print("y_knn.shape: ", y_knn.shape)
        knn_error_ratio = mean_absolute_percentage_error(y_test, y_knn)
        
        y_linear = y_linear.reshape(-1,1)
        y_linear = scaler.inverse_transform(y_linear)
        linear_error_ratio = mean_absolute_percentage_error(y_test, y_linear)
        
        y_decisionTree = y_decisionTree.reshape(-1,1)
        y_decisionTree = scaler.inverse_transform(y_decisionTree)
        decisionTree_error_ratio = mean_absolute_percentage_error(y_test, y_decisionTree)
        
        y_keras = y_keras.ravel().reshape(-1,1)
        y_keras = scaler.inverse_transform(y_keras)
        keras_error_ratio = mean_absolute_percentage_error(y_test, y_keras)
        
        y_randomForest = y_randomForest.reshape(-1,1)
        y_randomForest = y_randomForest.tolist()
        y_randomForest = scaler.inverse_transform(y_randomForest)
        randomForest_error_ratio = mean_absolute_percentage_error(y_test, y_randomForest)
        
        y_gbdt = y_gbdt.reshape(-1,1)
        y_gbdt = scaler.inverse_transform(y_gbdt)
        gbdt_error_ratio = mean_absolute_percentage_error(y_test, y_gbdt)
        
        print("keras: ", keras_error_ratio)
        print("knn:", "%.2f"%knn_error_ratio,"-svr:","%.2f"%svr_error_ratio, 
        "-linear:","%.2f"%linear_error_ratio,"-decisionTree:","%.2f"%decisionTree_error_ratio, "-randomForest:","%.2f"%randomForest_error_ratio,
        "-gbdt:","%.2f"%gbdt_error_ratio)
        
        #keras_error_sum = keras_error_sum + keras_error_ratio
        knn_error_sum = knn_error_sum + knn_error_ratio
        svr_error_sum = svr_error_sum + svr_error_ratio
        linear_error_sum = linear_error_sum + linear_error_ratio
        decisionTree_error_sum = decisionTree_error_sum + decisionTree_error_ratio
        randomForest_error_sum = randomForest_error_sum + randomForest_error_ratio
        gbdt_error_sum = gbdt_error_sum + gbdt_error_ratio
        keras_error_sum = keras_error_sum + keras_error_ratio
        
        #print("after 1 y_test.shape: ", y_test.shape)
        r = y_test.shape[0]
        
        y_results_i = np.zeros((0, y_test.shape[0]))
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_test).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_svr).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_knn).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_decisionTree).ravel(), axis=0) 
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_randomForest).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_gbdt).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_linear).ravel(), axis=0)
        y_results_i = np.insert(y_results_i, y_results_i.shape[0], values=np.array(y_keras).ravel(), axis=0)
        if y_results.shape[0] == 0:
            y_results = y_results_i
        else:
            y_results = y_results + y_results_i
    
    y_results = y_results/num
    
    avg_knn = knn_error_sum/num
    avg_svr = svr_error_sum/num
    avg_linear = linear_error_sum/num
    avg_decisionTree = decisionTree_error_sum/num
    avg_randomForest = randomForest_error_sum/num
    avg_gbdt = gbdt_error_sum/num
    avg_keras = keras_error_sum/num
    
    
    
    error_ratios = np.zeros((1,y_results.shape[1]-1))
    error_ratios[0,0] = avg_svr
    error_ratios[0,1] = avg_knn
    error_ratios[0,2] = avg_decisionTree
    error_ratios[0,3] = avg_randomForest
    error_ratios[0,4] = avg_gbdt
    error_ratios[0,5] = avg_linear
    error_ratios[0,6] = avg_keras
    
    #save_predict_data(pred_y=y_results, error_ratios=error_ratios, filename=outfile)
    print("finally------------------")
    print("knn:", "%.2f"%avg_knn,"-svr:","%.2f"%avg_svr, 
    "-linear:","%.2f"%avg_linear, "-decisionTree:","%.2f"%avg_decisionTree, "-randomForest:","%.2f"%avg_randomForest,
    "-gbdt:","%.2f"%avg_gbdt,"-keras:","%.2f"%avg_keras)
 
def getNewFeas(x, y, addNewFeasNum):
   new_x = np.zeros((0, x.shape[1]+addNewFeasNum))
   new_y = np.zeros((0,1))
   for i in range(len(y)):
       if i-addNewFeasNum < 0:
          continue
       feas = x[i,:].reshape(1,x.shape[1])
       new_fea = np.zeros((1, addNewFeasNum))
       for j in range(addNewFeasNum):
          new_fea[0,j]  = y[i-j-1]
       #print("feas.shape: ", feas.shape, " new_fea.shape: ", new_fea.shape)
       feas = np.append(feas,np.array(new_fea).reshape(1, addNewFeasNum), axis=1)
       new_x = np.append(new_x, feas.reshape(1,x.shape[1]+addNewFeasNum), axis=0)
       new_y = np.append(new_y, np.array(y[i]).reshape(1,-1), axis=0)
       #print("new_x i: ", new_x[new_x.shape[0]-1, :], " new_y i: ", new_y[new_y.shape[0]-1, :])
   print("x.shape: ", x.shape, " y.shape: ", y.shape, " new_x.shape: ", new_x.shape, " new_y.shape: ", new_y.shape)
   return new_x,new_y   
   
#根据y值，删除异常值对;
def deleteOutliers(x,y, low_thresh, high_thresh):
    min_value = min(y)
    max_value = max(y)

    new_x = x.copy()
    new_y = y.copy()
    delete_num = 0
    x_rows = x.shape[0]
    for i in range(x_rows):
        y_value = y[i]
        if y_value < low_thresh or y_value > high_thresh:
            new_x = np.delete(new_x, i-delete_num, axis=0)
            new_y = np.delete(new_y, i-delete_num, axis=0)
            delete_num = delete_num+1
    print("delete_num：", delete_num)
    return new_x,new_y     

def analizeData(y, thresh):
   num = 0
   for i in range(len(y)):
      if y[i] > thresh:
         num = num + 1
         
   return num*1.0/len(y)
   
#因为总氮=氨氮+销氮+亚销氮
#所以，总氮的值应该大于氨氮
def deleteTNOutliers(x_TN, y_TN, x_NH3N, y_NH3N):
    new_x_TN = x_TN.copy()
    new_y_TN = y_TN.copy()
    new_x_NH3N = x_NH3N.copy()
    new_y_NH3N = y_NH3N.copy()
    delete_num = 0
    x_rows = x_TN.shape[0]
    for i in range(x_rows):
        y_value = y_TN[i]
        if y_value < y_NH3N[i]:
            new_x_TN = np.delete(new_x_TN, i-delete_num, axis=0)
            new_y_TN = np.delete(new_y_TN, i-delete_num, axis=0)
            new_x_NH3N = np.delete(new_x_NH3N, i-delete_num, axis=0)
            new_y_NH3N = np.delete(new_y_NH3N, i-delete_num, axis=0)
            delete_num = delete_num+1
    print("delete_num：", delete_num)
    return new_x_TN,new_y_TN 
 
if __name__=='__main__': 

   print("*********load weather data***************")
   #读入天气数据
   weather_file = "./data/weather.xlsx"
   weather_feas, weather_time = load_weather_data(weather_file)  
   
   print("*********load TD sewage data***************")
   #读入进水数据
   sewage_file_list = "./data/TD.txt"
   datacols = 6
   isTD = True
   
   # #读入进水数据
   # sewage_file_list = "./data/ER.txt"
   # datacols = 5
   # isTD = False
   
   sewage_feas, sewage_time = load_sewage_data(sewage_file_list, isTD)
   
   print("*********align weather data and tuodong sewage data***************")
   #相同时间的天气数据与相同时间的进水数据对应起来

   align_time,x,y = align_weather_sewage(weather_feas, weather_time, sewage_feas, sewage_time, datacols)
   
   x = x.reshape(-1, x.shape[1])
   
   #'瞬时流量', '进水PH', '进水氨氮', '进水总磷', '进水COD', '进水总氮'
   #预测进水流量
   y_0 = y[:,0]
   addNewFeasNum = 5
   x_0, y_0 = getNewFeas(x, y_0, addNewFeasNum)
   
   #计算数据的pearson相关性
   for i in range(addNewFeasNum+2):
       r,p = stats.pearsonr(y_0.flatten(),x_0[:,i])
       print('进水流量与第0列特征 r: ', r, ' p: ', p)
   
   #预测进水PH
   y_1 = y[:,1]
   x_1, y_1 = getNewFeas(x, y_1, addNewFeasNum)
   for i in range(addNewFeasNum+2):
       r,p = stats.pearsonr(y_1.flatten(),x_1[:,i])
       print('pH与第0列特征 r: ', r, ' p: ', p)

   #进水氨氮
   y_2 = y[:,2]
   x_2, y_2 = getNewFeas(x, y_2, addNewFeasNum)
   for i in range(addNewFeasNum+2):
       r,p = stats.pearsonr(y_2.flatten(),x_2[:,i])
       print('进水氨氮与第0列特征 r: ', r, ' p: ', p)
       
   #进水总磷
   y_3 = y[:,3]
   x_3, y_3 = getNewFeas(x, y_3, addNewFeasNum)
   for i in range(addNewFeasNum+2):
       r,p = stats.pearsonr(y_3.flatten(),x_3[:,i])
       print('进水总磷与第0列特征 r: ', r, ' p: ', p)
       
   #进水COD
   y_4 = y[:,4]
   x_4, y_4 = getNewFeas(x, y_4, addNewFeasNum)
   for i in range(addNewFeasNum+2):
       r,p = stats.pearsonr(y_4.flatten(),x_4[:,i])
       print('进水总磷与第0列特征 r: ', r, ' p: ', p)
       
   #进水总氮 
   y_5 = y[:,5]  
   x_5, y_5 = getNewFeas(x, y_5, addNewFeasNum)
   
   
   y_0 = y_0.reshape(-1, 1)
   y_1 = y_1.reshape(-1, 1)
   y_2 = y_2.reshape(-1, 1)
   y_3 = y_3.reshape(-1, 1)
   y_4 = y_4.reshape(-1, 1)
   y_5 = y_5.reshape(-1, 1)
   
   #各浓度间的相关性
   r,p = stats.pearsonr(y_0.flatten(),y_1.flatten())
   print('进水流量与进水pH r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_0.flatten(),y_2.flatten())
   print('进水流量与进水氨氮 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_0.flatten(),y_3.flatten())
   print('进水流量与进水总磷 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_0.flatten(),y_4.flatten())
   print('进水流量与进水COD r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_1.flatten(),y_2.flatten())
   print('进水pH与进水氨氮 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_1.flatten(),y_3.flatten())
   print('进水pH与进水总磷 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_1.flatten(),y_4.flatten())
   print('进水pH与进水COD r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_2.flatten(),y_3.flatten())
   print('进水氨氮与进水总磷 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_2.flatten(),y_4.flatten())
   print('进水氨氮与进水COD r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_3.flatten(),y_4.flatten())
   print('进水总磷与进水COD r: ', r, ' p: ', p)
   
   r,p = stats.pearsonr(y_5.flatten(),y_0.flatten())
   print('进水总氮与进水流量 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_5.flatten(),y_1.flatten())
   print('进水总氮与进水pH r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_5.flatten(),y_2.flatten())
   print('进水总氮与进水氨氮 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_5.flatten(),y_3.flatten())
   print('进水总氮与进水总磷 r: ', r, ' p: ', p)
   r,p = stats.pearsonr(y_5.flatten(),y_4.flatten())
   print('进水总氮与进水COD r: ', r, ' p: ', p)
   
   # print("---------------predict influent-----------------------")
   # high_thresh = 1500
   # low_thresh = 50
   # x_0, y_0 = deleteOutliers(x_0, y_0, low_thresh, high_thresh)
   # min_y0 = min(y_0)
   # max_y0 = max(y_0)
   # print("min_y0: ", min_y0)
   # print("max_y0: ", max_y0)
   # outfile='out_influent'
   # print('X_0.shape: ', x_0.shape)
   # predict_MF(x_0, y_0, outfile)
   #predict(x_0, y_0, outfile)
  
   # print("-------------predict PH-----------------------")
   # high_thresh = 9
   # low_thresh = 5
   # x_1, y_1 = deleteOutliers(x_1, y_1, low_thresh, high_thresh)
   # min_y1 = min(y_1)
   # max_y1 = max(y_1)
   # print("min_y1: ", min_y1)
   # print("max_y1: ", max_y1)
   # outfile='out_pH'
   # predict_MF(x_1, y_1, outfile)
   # #predict(x_1, y_1, outfile)
   # print('X_1.shape: ', x_1.shape)
   # print("--------------predict 氨氮-----------------------")
   # high_thresh = 50
   # low_thresh = 1
   # x_2, y_2 = deleteOutliers(x_2, y_2, low_thresh, high_thresh)
   # min_y2 = min(y_2)
   # max_y2 = max(y_2)
   # print("min_y2: ", min_y2)
   # print("max_y2: ", max_y2)
   # outfile='out_NH3N'
   # predict_MF(x_2, y_2, outfile)
   # #predict(x_2, y_2, outfile)
   # print('X_2.shape: ', x_2.shape)
   # print("--------------predict 总磷-----------------------")
   # high_thresh = 10
   # low_thresh = 0.2
   # x_3, y_3 = deleteOutliers(x_3, y_3, low_thresh, high_thresh)
   # min_y3 = min(y_3)
   # max_y3 = max(y_3)
   # print("min_y3: ", min_y3)
   # print("max_y3: ", max_y3)
   # outfile='out_TP'
   # predict_MF(x_3, y_3, outfile)
   # #predict(x_3, y_3, outfile)
   # print('X_3.shape: ', x_3.shape)
   # print("---------------predict COD-----------------------")
   # high_thresh = 400
   # low_thresh = 20.0
   # x_4, y_4 = deleteOutliers(x_4, y_4, low_thresh, high_thresh)
   # min_y4 = min(y_4)
   # max_y4 = max(y_4)
   # print("min_y4: ", min_y4)
   # print("max_y4: ", max_y4)
   # outfile='out_COD'
   # #predict(x_4, y_4, outfile)
   # predict_MF(x_4, y_4, outfile)
   # print('X_4.shape: ', x_4.shape)
   # print("--------------predict 总氮-----------------------")
   # x_5, y_5 = deleteTNOutliers(x_5, y_5, x_2, y_2)
   # min_y5 = min(y_5)
   # max_y5 = max(y_5)
   # TN_thresh = 60
   # ratio_TN = analizeData(y_5, TN_thresh)
   # print("TN min_y5: ", min_y5, " max_y5: ", max_y5, " bigger than 60: ", ratio_TN)
   # low_thresh = 10
   # high_thresh = 90
   # x_5, y_5 = deleteOutliers(x_5, y_5, low_thresh, high_thresh)
   # outfile='out_TN'
   # #predict_MF(x_5, y_5, outfile)
   # #predict(x_5, y_5, outfile)
   # print('X_5.shape: ', x_5.shape)
