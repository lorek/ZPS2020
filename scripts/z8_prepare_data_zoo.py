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

# przyklad: python scripts/z7_prepare_data_haberman.py --input-dir datasets/haberman/ --output-dir datasets_prepared/haberman/

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
zoo_data = pd.read_csv(input_dir+"/zoo.csv")
zoo_data.drop('animal_name',axis=1,inplace=True)
#legs = pd.get_dummies(zoo_data['legs'],drop_first=True)
#zoo_data.drop('legs',axis=1,inplace=True)
#zoo_data = pd.concat([zoo_data,legs],axis=1)
X= zoo_data.drop('class_type', axis = 1)
y= zoo_data['class_type']


# nazwy klas 

classes_names = np.unique(y)
 

# dzielimy zbior na treningowy i testowy

x_train, x_test, y_train, y_test = train_test_split(X, y,
            test_size=0.30, random_state=42)
            

save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)