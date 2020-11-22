# z7
# prepare_data_connect4
#
# Przyklad: z katalogu --input-dir
# 
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl
#
# Wylicza macierz PCA na punktach train_data.pkl i je przeksztalca
# To samo przeksztalcenie wykonuje na train_data.pkl
# Wynik zapisywany jest do --output-dir


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

# Example:
# python scripts_learn/z7_prepare_data_connect4.py --input-dir datasets/connect4/ --output-dir datasets_prepared/connect4

 


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

# wczytaj plik data
df = pd.read_table(input_dir+"/connect4.data", sep=",")

# ta instrukcja mozna ogladnac format danych:
# print(df.head)

#integer encoding
df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: pd.factorize(x)[0])

# Zamieniamy DataFrame (df) na macierz numpy

data_all = df.to_numpy()

# pierwsze 42 kolumny to dane:

data = data_all[:,:-1] # = wszystkie wiersze, kolumny do 42.

# ostatnia kolumna to klasy:

data_classes = data_all[:,-1] # = wszystkie wiersze, kolumna ostatnia

# nazwy klas -- damy tutaj wszystkie unikalne numery, ktore wystepuja w data_classes
# (mozna np. reczni wpisac classes_names = ['divorced', 'not divorced']

classes_names = np.unique(data_classes)
 

# dzielimy zbior na treningowy i testowy

x_train, x_test, y_train, y_test = train_test_split(data, data_classes,
            test_size=test_size_fraction, random_state=42)
            

save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)
    
         


 
 
