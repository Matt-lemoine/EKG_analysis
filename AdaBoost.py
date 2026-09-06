from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv



def do_adaboost(X, y, learning_rate):

    model = AdaBoostClassifier(
        n_estimators=50, 
        learning_rate=learning_rate
    )

    auc_scores = cross_val_score(model, X, y, cv = 5, scoring='roc_auc')

    return auc_scores


Rns = ['R2', 'R3', 'R4'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# CatBoost parameters
learning_rates = [1, 0.5, 0.25]


learn_r = 0
while learn_r < len(learning_rates):

    learning_rate = learning_rates[learn_r]

    All_info_to_be_written = []

    r = 0
    while r < len(Rns):
        R_n = Rns[r]

        v = 0
        while v < len(Vectorizations):
            Vec_title = Vectorizations[v]

            el = 0
            while el < len(ending_leads):
                suffix_lead = ending_leads[el]

                if Vec_title == 'Betti_Vectorization':
                    i = 0
                    while i < len(Ns):
                        n = Ns[i]

                        csv_name = Vec_title + suffix_lead + '_' + str(n) + '_' + str(R_n) + '.csv'

                        file = f'Vectorizations_9_1_26/{csv_name}'

                        print(file)

                        dataset = pd.read_csv(file, header = None)
                        dataset = dataset.drop(dataset.columns[[0]], axis = 1)
                        max_col = dataset.shape[1]-1
                        X = dataset.iloc[:, 0:max_col]
                        y = dataset.iloc[:, max_col].values

                        auc_scores = do_adaboost(X, y, learning_rate)
                        mean_auc_score = auc_scores.mean()
                        std_auc_score = auc_scores.std()
                        
                        print(f'Average AUC over 5 fold cross validation for {file} with N = {n} is: {mean_auc_score}')

                        All_info_to_be_written.append([Vec_title + suffix_lead, R_n, mean_auc_score, std_auc_score])

                        i = i+1

                else:

                    csv_name = Vec_title + suffix_lead  + '_' + str(R_n) + '.csv'

                    file = f'Vectorizations_9_1_26/{csv_name}'

                    print(file)

                    dataset = pd.read_csv(file, header = 0)
                    dataset = dataset.drop(dataset.columns[[0]], axis = 1)
                    max_col = dataset.shape[1]-1
                    X = dataset.iloc[:, 0:max_col]
                    y = dataset.iloc[:, max_col].values

                    auc_scores = do_adaboost(X, y, learning_rate)
                    mean_auc_score = auc_scores.mean()
                    std_auc_score = auc_scores.std()

                    print(f'Average AUC over 5 fold cross validation for {file} is: {mean_auc_score}')

                    All_info_to_be_written.append([Vec_title + suffix_lead, R_n, mean_auc_score, std_auc_score])

                el = el+1
            
            v = v+1
        
        r = r+1

    All_info_to_be_written = np.array(All_info_to_be_written)

    output_csv_file = f"All_Results_AdaBoost_lr_{learning_rate}.csv"
    with open(output_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vectorization_Method_and_lead", "R_n", "Mean_AUC", "STD_AUC"])

        i = 0
        while i < len(All_info_to_be_written):
            row = All_info_to_be_written[i]
            writer.writerow(row)
            i = i+1

    print(f'################## Saved Results to {output_csv_file} ##################')

    learn_r = learn_r +1
