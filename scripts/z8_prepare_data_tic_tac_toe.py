from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import decomposition
 
from sklearn import datasets, metrics 
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
 
import time
import argparse

from sklearn.model_selection import train_test_split
import pickle 

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# przyklad: python scripts/z8_prepare_data_tic_tac_toe.py --input-dir datasets/tic_tac_toe/ --output-dir datasets_prepared/z8_tic_tac_toe/

def save_data(x_train , y_train, x_test , y_test, classes_names, output_dir):
    

    # zapisujemy dane treningowe
    x_train_all_dict  =    {     'data': x_train ,
                                    'classes':y_train}
                
    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
    pickle.dump(x_train_all_dict, train_data_outfile)



    # zapisujemy dane testowe 
    x_test_all_dict  =  {'data': x_test  ,
                        'classes':y_test}
     
    test_data_outfile = open(output_dir + '/test_data.pkl', 'wb')
    pickle.dump(x_test_all_dict, test_data_outfile)

    # zapisujemy nazwy klas
    cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
    pickle.dump(classes_names, cl_names_outfile)

    print("Pickles saved in ", output_dir)
    
    


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir (default: %(default)s)')
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--fraction', default="0.2", required=False, help='size of test set (fration) (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.fraction
        
input_dir, output_dir, test_size_fraction   = ParseArguments()

test_size_fraction = float(test_size_fraction)

# wczytujemy plik 
ttt_data = pd.read_table(input_dir+"/tic-tac-toe.data",sep=",", header=None) 
# robimy one-hot-encoding dla kazdej kolumny
cols0 = pd.get_dummies(ttt_data[0],drop_first=True)
cols1 = pd.get_dummies(ttt_data[1],drop_first=True)
cols2 = pd.get_dummies(ttt_data[2],drop_first=True)
cols3 = pd.get_dummies(ttt_data[3],drop_first=True)
cols4 = pd.get_dummies(ttt_data[4],drop_first=True)
cols5 = pd.get_dummies(ttt_data[5],drop_first=True)
cols6 = pd.get_dummies(ttt_data[6],drop_first=True)
cols7 = pd.get_dummies(ttt_data[7],drop_first=True)
cols8 = pd.get_dummies(ttt_data[8],drop_first=True)

#dodajemy do naszych danych i usuwamy stare kolumny
ttt_data.drop(0,axis=1,inplace=True)
ttt_data.drop(1,axis=1,inplace=True)
ttt_data.drop(2,axis=1,inplace=True)
ttt_data.drop(3,axis=1,inplace=True)
ttt_data.drop(4,axis=1,inplace=True)
ttt_data.drop(5,axis=1,inplace=True)
ttt_data.drop(6,axis=1,inplace=True)
ttt_data.drop(7,axis=1,inplace=True)
ttt_data.drop(8,axis=1,inplace=True)
ttt_data = pd.concat([ttt_data,cols0,cols1,cols2,cols3,cols4,cols5,cols6,cols7,cols8],axis=1)
#zmienilem nazwy kolumn zeby sie nie pomylic
ttt_data.columns = [9,'o0','x0','o1','x1','o2','x2','o3','x3','o4','x4','o5','x5','o6','x6','o7','x7','o8','x8']
#zapisujemy nasze dane 
X= ttt_data.drop(9, axis = 1) #usuwamy kolumne z klasyfikatorem i dostajemy nasze dane
y= ttt_data[9] #kolumna '9' to nasz klasyfikator z wartosciami positive negative


# nazwy klas 

classes_names = np.unique(y)
 

# dzielimy zbior na treningowy i testowy

x_train, x_test, y_train, y_test = train_test_split(X, y,
            test_size=0.20, random_state=42)
            

save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)
