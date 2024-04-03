# Importing Libraries
from collections import Counter
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import warnings
import time
from joblib import Parallel, delayed


app = Flask(__name__)
CORS(app)

# API Endpoint
@app.route('/predict_UAT', methods=['POST'])

def pred_disease():
    # Error messages
    start = time.time()
    if 'symptom1' not in request.json:
        return jsonify({'error': 'No symptom1 selected'}), 400
    elif 'symptom2' not in request.json:
        return jsonify({'error': 'No symptom2 selected'}), 400
    elif 'symptom3' not in request.json:
        return jsonify({'error': 'No symptom3 selected'}), 400
    # elif 'symptom4' not in request.form:
    #     return jsonify({'error': 'No symptom4 selected'}), 400

    # Retrieving data from webpage
    symptom1 = request.json['symptom1']
    symptom2 = request.json['symptom2']
    symptom3 = request.json['symptom3']

    nsymptoms = [symptom1, symptom2, symptom3]

    if 'symptom4' in request.json and request.json['symptom4'] != "":
        symptom4 = request.json['symptom4']
        nsymptoms.append(symptom4)

    if 'symptom5' in request.json and request.json['symptom5'] != "":
        symptom5 = request.json['symptom5']
        nsymptoms.append(symptom5)

    warnings.filterwarnings("ignore", message="X does not have valid feature names")




    # Symptoms List
    symp_list = pd.read_csv("Final_Dataset.csv", nrows=0).columns.tolist()
    symp_list.pop()
    print(symp_list)

    # Diseases List
    disease_list = ['Fungal Infection', 'Allergy', 'GERD', 'Chronic Cholestasis', 'Drug Reaction',
               'Peptic Ulcer Disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
               'Migraine', 'Cervical Spondylosis',
               'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chickenpox', 'Dengue', 'Typhoid', 'Hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic Hepatitis', 'Tuberculosis',
               'Common Cold', 'Pneumonia', 'Dimorphic Hemmorhoids (piles)',
               'Heart Attack', 'Varicose Veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthritis',
               'Arthritis', 'Vertigo', 'Acne', 'Urinary Tract Infection', 'Psoriasis',
               'Impetigo']

    # l2 = []
    # for x in range(0, len(symp_list)):
    #     l2.append(0)



# ------------------------------------------------ TRAINING DATA -------------------------------------------------------

    # Reading and preprocessing the training dataset csv file
    df = pd.read_csv("Final_Dataset.csv")
    # pd.set_option('future.no_silent_downcasting', True)
    # df.replace(
    #     {'prognosis': {'Fungal Infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic Cholestasis': 3, 'Drug Reaction': 4,
    #            'Peptic Ulcer Disease': 5, 'AIDS': 6, 'Diabetes': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension': 10,
    #            'Migraine': 11, 'Cervical Spondylosis': 12,
    #            'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chickenpox': 16, 'Dengue': 17, 'Typhoid': 18, 'Hepatitis A': 19,
    #            'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic Hepatitis': 24, 'Tuberculosis': 25,
    #            'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic Hemmorhoids (piles)': 28,
    #            'Heart Attack': 29, 'Varicose Veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthritis': 34,
    #            'Arthritis': 35, 'Vertigo': 36, 'Acne': 37, 'Urinary Tract Infection': 38, 'Psoriasis': 39,
    #            'Impetigo': 40}},
    #     inplace=True
    # )

    # print(df.head())

    X = df[symp_list]
    le = LabelEncoder()
    y = df[['prognosis']]
    y = le.fit_transform(y)

    y = np.ravel(y)

    print(X)
    print(y)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------ TESTING DATA -------------------------------------------------------

    # # Reading and preprocessing the testing dataset csv file
    # df2 = pd.read_csv("Testing.csv")
    # # pd.set_option('future.no_silent_downcasting', True)
    # # df2.replace(
    # #     {'prognosis': {'Fungal Infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic Cholestasis': 3, 'Drug Reaction': 4,
    # #            'Peptic Ulcer Disease': 5, 'AIDS': 6, 'Diabetes': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension': 10,
    # #            'Migraine': 11, 'Cervical Spondylosis': 12,
    # #            'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chickenpox': 16, 'Dengue': 17, 'Typhoid': 18, 'Hepatitis A': 19,
    # #            'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic Hepatitis': 24, 'Tuberculosis': 25,
    # #            'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic Hemmorhoids (piles)': 28,
    # #            'Heart Attack': 29, 'Varicose Veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthritis': 34,
    # #            'Arthritis': 35, 'Vertigo': 36, 'Acne': 37, 'Urinary Tract Infection': 38, 'Psoriasis': 39,
    # #            'Impetigo': 40}},
    # #     inplace=True
    # # )
    #
    # X_test = df2[symp_list]
    # y_test = df2[["prognosis"]]
    # le = LabelEncoder()
    # y_test = le.fit_transform(y_test)
    # y_test = np.ravel(y_test)

# ------------------------------------------------ DECISION TREE ------------------------------------------------------

    def DecisionTree():

        clf3 = tree.DecisionTreeClassifier()
        clf3 = clf3.fit(X_train, y_train)

        # Training dataset and calculating accuracy
        y_pred = clf3.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

        # Predicting the disease
        inputtest = [l2]
        predict = clf3.predict(inputtest)
        predicted = predict[0]

        nf = "Not Found"
        h = "no"
        b = 0
        for a in range(0, len(disease_list)):
            if predicted == a:
                b = a
                h = "yes"
                break

        if h == "yes":
            return disease_list[b]
        else:
            return nf
        
# ------------------------------------------------ RANDOM FOREST ------------------------------------------------------

    def RandomForest():

        clf4 = RandomForestClassifier()
        clf4 = clf4.fit(X_train, y_train)

        # Training dataset and calculating accuracy
        y_pred = clf4.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

        # Predicting the disease
        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted = predict[0]

        nf = "Not Found"
        h = "no"
        b = 0
        for a in range(0, len(disease_list)):
            if predicted == a:
                b = a
                h = "yes"
                break

        if h == "yes":
            return disease_list[b]
        else:
            return nf

# ------------------------------------------------ NAIVE BAYES ------------------------------------------------------

    def NaiveBayes():

        clf5 = GaussianNB()
        clf5 = clf5.fit(X_train, y_train)

        # Training dataset and calculating accuracy
        y_pred = clf5.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

        # Predicting the disease
        inputtest = [l2]
        predict = clf5.predict(inputtest)
        predicted = predict[0]

        nf = "Not Found"
        h = "no"
        b = 0
        for a in range(0, len(disease_list)):
            if predicted == a:
                b = a
                h = "yes"
                break

        if h == "yes":
            return disease_list[b]
        else:
            return nf

# ------------------------------------------------ SUPPORT VECTOR MACHINES (SVM) ---------------------------------------
    def SVM():

        clf6 = SVC(kernel='rbf', gamma='auto')
        clf6.fit(X_train, y_train)

        # Training dataset and calculating accuracy
        y_pred = clf6.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

                # Predicting the disease
                inputtest = [l2]
                predict = clf6.predict(inputtest)
                predicted = predict[0]

                nf = "Not Found"
                h = "no"
                b = 0
                for a in range(0, len(disease_list)):
                    if predicted == a:
                        b = a
                        h = "yes"
                        break

                if h == "yes":
                    return disease_list[b]
                else:
                    return nf

# ------------------------------------------------ K-NEAREST NEIGHBORS (KNN) ------------------------------------------------------
    def KNN():
        clf7 = KNeighborsClassifier()  # Specify the KNN classifier, you can change the number of neighbors as needed
        param_grid = {
            'n_neighbors': [3, 5, 7],  # Number of neighbors
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'metric': ['euclidean', 'manhattan']  # Distance metric
        }
        # Perform grid search
        grid_search = GridSearchCV(estimator=clf7, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best parameters from grid search
        best_params = grid_search.best_params_

        # Train KNN with the best parameters
        best_knn = KNeighborsClassifier(**best_params)
        best_knn.fit(X, y)

        # Training dataset and calculating accuracy
        y_pred = best_knn.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

                # Predicting the disease
                inputtest = [l2]
                predict = best_knn.predict(inputtest)
                predicted = predict[0]

                nf = "Not Found"
                h = "no"
                b = 0
                for a in range(0, len(disease_list)):
                    if predicted == a:
                        b = a
                        h = "yes"
                        break

                if h == "yes":
                    return disease_list[b]
                else:
                    return nf


# ------------------------------------------------ GRADIENT BOOSTING MACHINES (GBM) ------------------------------------------------------
    def GBM():

        clf8 = GradientBoostingClassifier()  

        # Define hyperparameters grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of boosting stages
            'learning_rate': [0.05, 0.1, 0.2],  # Learning rate
            'max_depth': [3, 4, 5]  # Maximum depth of individual trees
        }

        # Perform grid search
        grid_search = GridSearchCV(estimator=clf8, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X, y)

        # Get the best parameters from grid search
        best_params = grid_search.best_params_

        # Train KNN with the best parameters
        best_gbm = GradientBoostingClassifier(**best_params)
        best_gbm.fit(X_train, y_train)

        # Training dataset and calculating accuracy
        y_pred = best_gbm.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=True))

        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)

                # Predicting the disease
                inputtest = [l2]
                predict = best_gbm.predict(inputtest)
                predicted = predict[0]

                nf = "Not Found"
                h = "no"
                b = 0
                for a in range(0, len(disease_list)):
                    if predicted == a:
                        b = a
                        h = "yes"
                        break

                if h == "yes":
                    return disease_list[b]
                else:
                    return nf
                
# ------------------------------------------------ Cat Boost Classifier ------------------------------------------------------

    def CBC():

        # Define CatBoost classifier
        loaded_model = CatBoostClassifier()
        loaded_model.load_model('catboost_model.bin')



        # Posting flags for the symptoms and storing them in a list
        # for k in range(0, len(symp_list)):
        #     for z in nsymptoms:
        #         if z == symp_list[k]:
        #             l2[k] = 1

        l2 = []

        for i in range(0, len(symp_list)):
            flag = "No"
            for j in range(0, len(nsymptoms)):
                if nsymptoms[j] == symp_list[i]:
                    l2.append(1)
                    flag = "Yes"
            if flag == "No":
                l2.append(0)


        # Predicting the disease
        inputtest = [l2]
        predict = loaded_model.predict(inputtest)
        predicted = predict[0]

        nf = "Not Found"
        h = "no"
        b = 0
        for a in range(0, len(disease_list)):
            if predicted == a:
                b = a
                h = "yes"
                break

        if h == "yes":
            return disease_list[b]
        else:
            return nf



# ------------------------------------------------ RESULT OUTPUT ------------------------------------------------------

    dt_pred1 = DecisionTree()
    dt_pred2 = DecisionTree()
    dt_pred3 = DecisionTree()
    rf_pred1 = RandomForest()
    rf_pred2 = RandomForest()
    rf_pred3 = RandomForest()
    nb_pred1 = NaiveBayes()
    nb_pred2 = NaiveBayes()
    nb_pred3 = NaiveBayes()
    # sv_pred = SVM()
    # gb_pred = GBM()
    # kn_pred = KNN()
    cb_pred = CBC()


    predictions = [dt_pred1, dt_pred2, dt_pred3, rf_pred1, rf_pred2, rf_pred3, nb_pred1, nb_pred2, nb_pred3, cb_pred]

    # Count occurrences of each prediction
    prediction_counts = Counter(predictions)

    # Calculate percentages
    total_predictions = len(predictions)
    percentages = {prediction: (count / total_predictions) * 100 for prediction, count in prediction_counts.items()}

    # Format results into JSON

    # return jsonify({'Decision Tree': dt_pred,
    #                 'Random Forest': rf_pred,
    #                 'Naive Bayes': nb_pred,})

    print("----------------------------------------------------")
    print("Total processing time: ", time.time() - start)
    return jsonify({'predictions': predictions, 'percentages': percentages, 'symptoms': nsymptoms})

# ------------------------------------------------ HOSTING API ------------------------------------------------------
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    app.run(host=host, port=port, debug=True)