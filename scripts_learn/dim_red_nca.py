#
# Przyklad: z katalogu --input-dir
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl
#
# Wylicza macierz NCA na punktach train_data.pkl i je przeksztalca
# To samo przeksztalcenie wykonuje na train_data.pkl
# Wynik zapisywany jest do --output-dir



import numpy as np
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
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC

# Example:
# python scripts_learn/dim_red_nca.py --n 3 --input-dir datasets_prepared/mnist_test1/ 
#                                       --output-dir datasets_prepared/mnist_test1_nca3d/



def read_data(input_dir):
    # wczytujemy dane treningowe:
    train_data_infile = open(input_dir + '/train_data.pkl', 'rb')  # czytanie z pliku
    data_train_all_dict =  pickle.load(train_data_infile)

    x_train = data_train_all_dict["data"]
    y_train = data_train_all_dict["classes"]

    # wczytujemy dane testowe:
    test_data_infile = open(input_dir + '/test_data.pkl', 'rb')  # czytanie z pliku
    data_test_all_dict =  pickle.load(test_data_infile)

    x_test= data_test_all_dict["data"]

    y_test = data_test_all_dict["classes"]

    # i nazwy klas 
    cl_names_infile = open(input_dir + '/class_names.pkl', 'rb')
    classes_names =  pickle.load(cl_names_infile)

    print("Data loaded from " + input_dir)
    
    return x_train, y_train, x_test, y_test, classes_names


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
    parser.add_argument('--n', default=None, required=False, help='dimension of reduction')    
    parser.add_argument('--max_iter', default=50, required=False, help='aproximately number of neighbours')   
    parser.add_argument('--random_state', default=None, required=False, help='seed')
    parser.add_argument('--init', default='auto', required=False, help='default random alternatively pca')   
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.n, args.max_iter, args.random_state, args.init
        
input_dir, output_dir, n, max_iter, random_state, init = ParseArguments()
    
if n != None:
    n = int(n)
    
max_iter = int(max_iter)

if random_state != None:
    random_state = int(random_state)   
    
# wczytujemy dane 
x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)
 


### NCA

print("NCA reduction ", x_train.shape[1], " -> ", n,  " ...",  end =" ")



nca = NeighborhoodComponentsAnalysis(n_components = n, max_iter=max_iter, random_state=random_state, init=init)

## wyuczenie macierzy tranformacji i zaaplikowanie jej na x_train
start_time = time.time()
x_train_reduced =  nca.fit_transform(x_train,y_train)
print("  took %s seconds " % round((time.time() - start_time),5))

#zastosowanie tej samej macierzy na x_test

x_test_reduced = nca.transform(x_test)


# zapisujemy dane 

save_data(x_train_reduced, y_train, x_test_reduced, y_test, classes_names, output_dir)
