#
# logreg.py
#
# Przyklad: z katalogu --input_dir
# trzy pliki: train_data.pkl, test_data.pkl, class_names.pkl
#
# Uczy sie na train_data.pkl, przewiduje na test_data.pkl


# Example
# python scripts/logreg.py --input-dir datasets_prepared/zoo --solver liblinear --C 10 --max_iter 400


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
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

 
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
 
def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='input dir (default: %(default)s)')
    parser.add_argument('--solver', default="lbfgs", required=False, help='solver: newton-cg, lbfgs, liblinear, sag, saga (default: %(default)s)')
    parser.add_argument('--C', default=1.0, required=False, help = 'C = 1.0 (default: %(default)s)')
    parser.add_argument('--max_iter', default=100,required=False, help = 'max_iter = 100 (default: %(default)s)' )
    args = parser.parse_args()

    return args.input_dir, args.solver, args.C, args.max_iter
        
input_dir, solv, cc, max_it = ParseArguments()
cc = float(cc)
solv = str(solv)
max_it = int(max_it)   
x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)


# wyswietlmy ile punktow jakiej klasy jest 
print("\n")
print("General info on train and test set:")
print("x_train, nr of points: ", x_train.shape[0],", dimension = ", x_train.shape[1])
print("x_test, nr of points: ", x_test.shape[0],", dimension = ", x_test.shape[1])

for i in np.unique(y_train):
    print("class ", i , ", train set size: ", len(y_train[y_train==i]), ", test set size: ", len(y_test[y_test==i]))


print("\n")
### KLASYFIKACJA

# definiujemy klasyfikator (logreg)
logreg = LogisticRegression(solver=solv, C=cc,max_iter=max_it)

# "uczymy" sie na zbiorze treningowym
start_time = time.time()
print("Learning and predicting with logreg ...",  end =" ")
logreg.fit(x_train, y_train)

# przewidujemy na testowym
y_pred =logreg.predict(x_test)
print("  took %s seconds " % round((time.time() - start_time),5))
    
# na testowym znalismy prawdziwe klasy, mozemy porownac jak "dobrze" poszlo
# (rozne metryki, tutaj przyklad

metric_accuracy = metrics.accuracy_score(y_test,y_pred)
print("logreg: accuracy = ", metric_accuracy)

 
# dla binarnych:
#metric_f1 = metrics.f1_score(y_test,y_pred)
#metric_precision = metrics.precision_score(y_test,y_pred)
#metric_recall = metrics.recall_score(y_test,y_pred)


print("full classification report:")
if type(classes_names) is not list:
    target_nms = classes_names.astype(str)
else:
    target_nms = classes_names
    
print(classification_report(y_test, y_pred,target_names = target_nms))
