# z5 chisquared feature selection

from sklearn.feature_selection import chi2
import numpy

import pandas as pd

import argparse

import pickle
# Example:
# python scripts/z5_chisquared.py --input-dir datasets_prepared/oral_toxicity --output-dir datasets_prepared/chisquared/oral_toxicity_3d --attributes 3

# parametr attributes mówi nam do ilu kolumn chcemy zredukować nasze dane
import time
import os


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir (default: %(default)s)')
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--attributes', default="1", required=False,
                        help='number of attributes to remain (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.attributes

def get_best_variables_index(n, X, y):
    odp = []
    ch = chi2(X, y)[1]
    pom = ch
    pom = numpy.sort(pom)[:n]
    for i in range(len(ch)):
        for j in range(len(pom)):
            if pom[j] == ch[i]:
                odp.append(i)
    return odp

def new_array(index, X):
    odp = X.take(index)

    return odp

input_dir, output_dir, attributes = ParseArguments()

attributes = int(attributes)


test = pd.read_pickle(input_dir + "/test_data.pkl")

train = pd.read_pickle(input_dir + "/train_data.pkl")

classes_names = pd.read_pickle(input_dir + "/class_names.pkl")

x_train = train['data']
x_test = test['data']

y_train = train['classes']
y_test = test['classes']

print("Twoje dane są o wymiarze {}".format(x_test.shape[1]))
start_time = time.time()

index = get_best_variables_index(attributes, x_train, y_train)
x_train_new = new_array(index, x_train)
x_test_new = new_array(index, x_test)
y_train_new = y_train.copy()
y_test_new = y_test.copy()

print("Redukcja wymiarów zajęła {} sekund".format(round(time.time() - start_time,5)))

def save_data(x_train, y_train, x_test, y_test, output_dir,attributes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    x_train_all_dict = {'data': x_train,
                        'classes': y_train}
    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb+')
    pickle.dump(x_train_all_dict, train_data_outfile)


    x_test_all_dict = {'data': x_test,
                       'classes': y_test}
    test_data_outfile = open(output_dir+ '/test_data.pkl', 'wb+')
    pickle.dump(x_test_all_dict, test_data_outfile)

    cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
    pickle.dump(classes_names, cl_names_outfile)

    print("Pickles saved in", output_dir,"with %d columns" % attributes)

save_data(x_train_new, y_train_new, x_test_new, y_test_new, output_dir,attributes)