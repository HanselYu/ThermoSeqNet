import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost
import lightgbm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
import glob
import os


def Load_features():
    Ifeature = np.array(pd.read_excel('Features_All_Accurate.xlsx', index_col=0))
    features_packed = (Ifeature[:, :1100],
                       Ifeature[:, 1100:1100 + 1024],
                       Ifeature[:, 1100 + 1024:1100 + 1024+1536],
                       Ifeature[:, 1100 + 1024+1536:])
    Label = []
    with open('Enzyme_Sequence.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] == '>':
                Label.append(float(line[line.index('_')+1:-1]))
    Label = np.array(Label)
    return features_packed, Label


def data_transfomation(features_packed_test):
    Mydir = sorted(glob.glob('ThermoSeq_d1.0/First_Model/*.pkl'))
    x_test_pre = []
    i = 0
    for dir in Mydir:
        print(i, dir)
        model = joblib.load(dir)
        x_test_pro = model.predict_proba(features_packed_test[int(dir[27]) - 1])
        x_test_pre.append(x_test_pro)
        i += 1
    x_test_pre = np.array(x_test_pre)
    x_test_pre = np.transpose(x_test_pre, (2, 0, 1))[1]
    x_test_pre = np.transpose(x_test_pre, (1, 0))
    print(x_test_pre.shape)
    return x_test_pre


def Independent_test(features_packed_ind_test, Ind_Label):
    x_test_pre = data_transfomation(features_packed_ind_test)
    Predicted_label = []
    model = joblib.load('ThermoSeq_d1.0/Second_Model/1_.h5')
    Pre_label = model.predict(x_test_pre)
    Pre_label = model.predict_proba(x_test_pre)
    for i in range(len(Pre_label)):
        Predicted_label.append(Pre_label[i][1])
    Predicted_label = np.array(Predicted_label)
    res = pd.DataFrame({'Predicted_thermo_prob': Predicted_label, 'Real_opt': Ind_Label})
    res.to_excel('ThermoSeq_c1.0/Proteome_predict_brenda.xlsx')


if __name__ == '__main__':
    features_packed, Label = Load_features()
    Independent_test(features_packed, Label)
