import numpy as np
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def save_data(x_train, y_train, x_test, y_test, classes_names, output_dir):
    # zapisujemy dane treningowe
    x_train_all_dict = {'data'   : x_train,
                        'classes': y_train}

    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
    pickle.dump(x_train_all_dict, train_data_outfile)

    # zapisujemy dane testowe
    x_test_all_dict = {'data'   : x_test,
                       'classes': y_test}

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
    parser.add_argument('--fraction', default="0.2", required=False,
                        help='size of test set (fration) (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.fraction


input_dir, output_dir, test_size_fraction = ParseArguments()

test_size_fraction = float(test_size_fraction)

# wczytaj plik nursery.data
df = pd.read_table(input_dir + "/nursery.data", sep=",", header=None)

# JEST 9 kolumn, ostatnia to klasyfikator

# Integer encoding zachowujący porządek danych 
int_enc = {0: {'usual': 0, 'pretentious': 1, 'great_pret': 2},
           1: {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4},
           2: {'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3},
           3: {'more': 3},
           4: {'convenient': 0, 'less_conv': 1, 'critical': 2},
           5: {'convenient': 0, 'inconv': 1},
           6: {'nonprob': 0, 'slightly_prob': 1, 'problematic': 2},
           7: {'recommended': 0, 'priority': 1, 'not_recom': 2},
           8: {'not_recom': 0, 'recommend': 1, 'very_recom': 2, 'priority': 3, 'spec_prior': 4}}
df = df.replace(int_enc)

# Zamieniamy DataFrame (df) na macierz numpy

data_all = df.to_numpy().astype(int)

# pierwsze 8 kolumn to dane:

data = data_all[:, :8]  # = wszystkie wiersze, kolumny do 8.

# ostatnia kolumna to klasy:

data_classes = data_all[:, 8]  # = wszystkie wiersze, kolumna 9

# nazwy klas -- damy tutaj wszystkie unikalne numery, ktore wystepuja w data_classes

classes_names = np.unique(data_classes)

# dzielimy zbior na treningowy i testowy

x_train, x_test, y_train, y_test = train_test_split(data, data_classes,
                                                    test_size=test_size_fraction, random_state=42)

save_data(x_train, y_train, x_test, y_test, classes_names, output_dir)
