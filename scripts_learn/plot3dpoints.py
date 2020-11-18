from sklearn import svm
import pickle
import argparse
import os


#
 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
  
 

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
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--input-dir', default="", required=True, help='input dir (default: %(default)s)')
    args = parser.parse_args()

    return  args.input_dir 

input_dir = ParseArguments()


x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)


 

if x_train.shape[1]!=3:
    print("These are not 3d points !!! (they are "+str(x_train.shape[1])+"-dimensional)")
    quit()

#osobne rysunki dla train i test

#TRAIN
fig_train = plt.figure(1)
ax_train = fig_train.add_subplot(111, projection='3d')

ax_train.set_title(input_dir +" - TRAIN")	

for nr, cl in enumerate(np.sort(np.unique(y_train))):
    points = x_train[y_train==cl];
    ax_train.scatter(points[:,0], points[:,1], points[:,2], label=classes_names[nr])
    
ax_train.legend()



fig_test = plt.figure(2)
ax_test = fig_test.add_subplot(111, projection='3d')

ax_test.set_title(input_dir +" - TEST")	

for nr, cl in  enumerate(np.sort(np.unique(y_test))):
    points = x_test[y_test==cl];
    ax_test.scatter(points[:,0], points[:,1], points[:,2], label=classes_names[nr])
    
ax_test.legend()


 
    
 
 
plt.show()

quit()


