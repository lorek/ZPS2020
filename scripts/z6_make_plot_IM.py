from sklearn import datasets, metrics, feature_selection
from functools import partial
import time
import pickle
from pylab import plot, show, figure

def read_data(input_dir):
    # wczytujemy dane treningowe:
    train_data_infile = open(input_dir + '/train_data.pkl', 'rb')  # czytanie z pliku
    data_train_all_dict = pickle.load(train_data_infile)

    x_train = data_train_all_dict["data"]
    y_train = data_train_all_dict["classes"]

    # wczytujemy dane testowe:
    test_data_infile = open(input_dir + '/test_data.pkl', 'rb')  # czytanie z pliku
    data_test_all_dict = pickle.load(test_data_infile)

    x_test = data_test_all_dict["data"]

    y_test = data_test_all_dict["classes"]

    # i nazwy klas
    cl_names_infile = open(input_dir + '/class_names.pkl', 'rb')
    classes_names = pickle.load(cl_names_infile)

    print("Data loaded from " + input_dir)

    return x_train, y_train, x_test, y_test, classes_names


def save_data(x_train, y_train, x_test, y_test, classes_names, output_dir):
    # zapisujemy dane treningowe
    x_train_all_dict = {'data': x_train,
                        'classes': y_train}

    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
    pickle.dump(x_train_all_dict, train_data_outfile)

    # zapisujemy dane testowe
    x_test_all_dict = {'data': x_test,
                       'classes': y_test}

    test_data_outfile = open(output_dir + '/test_data.pkl', 'wb')
    pickle.dump(x_test_all_dict, test_data_outfile)

    # zapisujemy nazwy klas
    cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
    pickle.dump(classes_names, cl_names_outfile)

    print("Pickles saved in ", output_dir)


input_dir = input("Adres pliku:")

x_train,y_train, x_test,y_test,classes_names = read_data(input_dir)

print("Twoje dane mają rozmiar:",x_train.shape[1])

n_comp = 3

fs = feature_selection.SelectKBest(score_func=partial(feature_selection.mutual_info_classif, n_neighbors=3), k=int(n_comp))
start_time = time.time()
x_train_reduced = fs.fit_transform(x_train,y_train)
print("  took %s seconds " % round((time.time() - start_time),5))
x_test_reduced = fs.transform(x_test)


new = x_test_reduced
fig = figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(new[:,0], new[:,1], new[:,2], c=y_test, marker ="x")
show()

# C:/Users/zuzbrzo/studia/sem5/zps/ZPS2020/datasets_prepared/teaching_assistant_evaluation
# C:/Users/zuzbrzo/studia/sem5/zps/ZPS2020/datasets_prepared/wine