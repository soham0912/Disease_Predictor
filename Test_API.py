# Importing Libraries
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# API Endpoint
@app.route('/predict-disease', methods=['POST'])

def pred_disease():

    # Error messages
    if 'symptom1' not in request.form:
        return jsonify({'error': 'No symptom1 selected'}), 400
    elif 'symptom2' not in request.form:
        return jsonify({'error': 'No symptom2 selected'}), 400
    elif 'symptom3' not in request.form:
        return jsonify({'error': 'No symptom3 selected'}), 400
    elif 'symptom4' not in request.form:
        return jsonify({'error': 'No symptom4 selected'}), 400

    # Retrieving data from webpage
    symptom1 = request.form['symptom1']
    symptom2 = request.form['symptom2']
    symptom3 = request.form['symptom3']
    symptom4 = request.form['symptom4']

    nsymptoms = [symptom1, symptom2, symptom3, symptom4]

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
    disease_list = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
               'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
               ' Migraine', 'Cervical spondylosis',
               'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
               'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
               'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
               'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
               'Impetigo']

    l2 = []
    for x in range(0, len(symp_list)):
        l2.append(0)

# ------------------------------------------------ TRAINING DATA -------------------------------------------------------

    # Reading and preprocessing the training dataset csv file
    df = pd.read_csv("Training.csv")
    # pd.set_option('future.no_silent_downcasting', True)
    df.replace(
        {'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                       'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
                       'Hypertension ': 10,
                       'Migraine': 11, 'Cervical spondylosis': 12,
                       'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                       'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                       'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                       'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                       'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                       'Varicose veins': 30, 'Hypothyroidism': 31,
                       'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                       '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                       'Psoriasis': 39,
                       'Impetigo': 40}},
        inplace=True
    )

    # print(df.head())

    X = df[symp_list]
    y = df[["prognosis"]]
    np.ravel(y)

# ------------------------------------------------ TESTING DATA -------------------------------------------------------

    # Reading and preprocessing the testing dataset csv file
    df2 = pd.read_csv("Testing.csv")
    # pd.set_option('future.no_silent_downcasting', True)
    df2.replace(
        {'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                       'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
                       'Hypertension ': 10,
                       'Migraine': 11, 'Cervical spondylosis': 12,
                       'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                       'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                       'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                       'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                       'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                       'Varicose veins': 30, 'Hypothyroidism': 31,
                       'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                       '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                       'Psoriasis': 39,
                       'Impetigo': 40}}, inplace=True)

    X_test = df2[symp_list]
    y_test = df2[["prognosis"]]
    np.ravel(y_test)

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
# ------------------------------------------------ RESULT OUTPUT ------------------------------------------------------

    dt_pred = DecisionTree()
    rf_pred = RandomForest()
    nb_pred = NaiveBayes()
    return jsonify({'Decision Tree': dt_pred,
                    'Random Forest': rf_pred,
                    'Naive Bayes': nb_pred})

# ------------------------------------------------ HOSTING API ------------------------------------------------------
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    app.run(host=host, port=port, debug=True)