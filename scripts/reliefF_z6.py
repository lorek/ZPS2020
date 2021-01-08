import numpy as np
import time
import argparse
from skrebate import ReliefF
import pickle



# Example:
# python scripts/reliefF_z6.py --n-neighbors 5  --n-features 3 --input-dir datasets_prepared/wine
#                                       --output-dir datasets_prepared/wine_reliefF3d



def read_data(input_dir):
    # wczytujemy dane treningowe:
    train_data_infile = open(input_dir + '/train_data.pkl', 'rb')  # czytanie z pliku
    data_train_all_dict = pickle.load(train_data_infile)

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
    parser.add_argument('--n-features', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--n-neighbors', default="", required=True, help='output dir (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir,args.output_dir, args.n_features, args.n_neighbors

input_dir, output_dir, n_features, n_neighbors  = ParseArguments()
n_features = int(n_features)
n_neighbors = int(n_neighbors)

# wczytujemy dane
x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)



### ReliefF

print("reliefF feature selection", x_train.shape[1], " -> ", n_features,  " ...",  end =" ")



reliefF = ReliefF(n_features_to_select= n_features, n_neighbors= n_neighbors)

## wybranie odpowiednich atrybutów na danych uczących
start_time = time.time()
x_train_reduced = reliefF.fit_transform(x_train, y_train)
print("  took %s seconds " % round((time.time() - start_time),5))

##wyciągniecie jedynie tych atrybutów z danych testowych

x_test_reduced = reliefF.transform(x_test)

# zapisujemy dane

save_data(x_train_reduced, y_train, x_test_reduced, y_test, classes_names, output_dir)

# dodatkowo zapisujemy sam obiekt reliefF
reliefF_object_file = open(output_dir+"/reliefF_object.pkl","wb")
pickle.dump(reliefF,reliefF_object_file)

