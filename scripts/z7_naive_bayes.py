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

from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
#from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import LabelEncoder


# przyklad: python scripts/z7_naive_bayes.py --input-dir datasets_prepared/haberman --data-type cat

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
    parser.add_argument('--input-dir', default="", required=True,
                        help='input dir (default: %(default)s)')
    parser.add_argument('--data-type', default="cat", required=False,
                        help='data type - possible types: cat(categorical), cont(continuous), bern(Bernoulli),\
                        multin(multinomial), cant-cont, (default: %(default)s)')
    parser.add_argument('--disc-col', default="", required=False,
                        help='discrete columns numbers (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.data_type, args.disc_col
        
input_dir, data_type, disc_col = ParseArguments()
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

## naive_bayes
# definiujemy klasyfikator(w zaleznosci od tego co podalismy w data-type):
if data_type == 'cat':
    n_b = CategoricalNB()
elif data_type == 'cont':
    n_b = GaussianNB()
elif data_type == 'bern':
    n_b = BernoulliNB()
elif data_type == 'multin':
    n_b = MultinomialNB()
'''
elif data_type == 'mixed':
    temp_list = disc_col.split(sep=",")
    disc_columns = [int(i) for i in temp_list]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    label_encoder = LabelEncoder()
    for i in disc_columns:
        x_train[i] = label_encoder.fit_transform(x_train[i])
        x_test[i] = label_encoder.fit_transform(x_test[i])
    n_b = MixedNB(categorical_features=disc_columns)

'''


# "uczymy" sie na zbiorze treningowym
start_time = time.time()
print("Learning and predicting with naive_bayes ...",  end =" ")
n_b.fit(x_train, y_train)




# przewidujemy na testowym
y_pred = n_b.predict(x_test)
print("  took %s seconds " % round((time.time() - start_time),5))
    
# na testowym znalismy prawdziwe klasy, mozemy porownac jak "dobrze" poszlo

metric_accuracy = metrics.accuracy_score(y_test,y_pred)
print("naive_bayes: accuracy = ", metric_accuracy)

print("full classification report:")
if type(classes_names) is not list:
    target_nms = classes_names.astype(str)
else:
    target_nms = classes_names
    
print(classification_report(y_test, y_pred, target_names = target_nms))     

########################################

