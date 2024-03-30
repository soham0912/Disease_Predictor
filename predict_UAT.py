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
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import time


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
    symp_list = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
          'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
          'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
          'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
          'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
          'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
          'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
          'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
          'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
          'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
          'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
          'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
          'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
          'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
          'family_history', 'mucoid_sputum',
          'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
          'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
          'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
          'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
          'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
          'yellow_crust_ooze']

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

    l2 = []
    for x in range(0, len(symp_list)):
        l2.append(0)

# ------------------------------------------------ TRAINING DATA -------------------------------------------------------

    # Reading and preprocessing the training dataset csv file
    df = pd.read_csv("Training.csv")
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

# ------------------------------------------------ TESTING DATA -------------------------------------------------------

    # Reading and preprocessing the testing dataset csv file
    df2 = pd.read_csv("Testing.csv")
    # pd.set_option('future.no_silent_downcasting', True)
    # df2.replace(
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

    X_test = df2[symp_list]
    y_test = df2[["prognosis"]]
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_test = np.ravel(y_test)

# ------------------------------------------------ DECISION TREE ------------------------------------------------------

    def DecisionTree():

        clf3 = tree.DecisionTreeClassifier()
        clf3 = clf3.fit(X, y)

        # Training dataset and calculating accuracy
        y_pred = clf3.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

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
        clf4 = clf4.fit(X, np.ravel(y))

        # Training dataset and calculating accuracy
        y_pred = clf4.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

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
        clf5 = clf5.fit(X, np.ravel(y))

        # Training dataset and calculating accuracy
        y_pred = clf5.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

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

        clf6 = SVC(kernel='linear')
        clf6.fit(X, y)

        # Training dataset and calculating accuracy
        y_pred = clf6.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

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
        clf7 = KNeighborsClassifier(n_neighbors=5)  # Specify the KNN classifier, you can change the number of neighbors as needed
        clf7.fit(X, y)

        # Training dataset and calculating accuracy
        y_pred = clf7.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

                # Predicting the disease
                inputtest = [l2]
                predict = clf7.predict(inputtest)
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
        clf8 = GradientBoostingClassifier()  # Specify the GBM classifier, you can tune hyperparameters as needed
        clf8.fit(X, y)

        # Training dataset and calculating accuracy
        y_pred = clf8.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))

        # Posting flags for the symptoms and storing them in a list
        for k in range(0, len(symp_list)):
            for z in nsymptoms:
                if z == symp_list[k]:
                    l2[k] = 1

                # Predicting the disease
                inputtest = [l2]
                predict = clf8.predict(inputtest)
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
    sv_pred = SVM()
    gb_pred = GBM()
    kn_pred = KNN()


    predictions = [dt_pred1, dt_pred2, dt_pred3, rf_pred1, rf_pred2, rf_pred3, nb_pred1, nb_pred2, nb_pred3, sv_pred, gb_pred, kn_pred]

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