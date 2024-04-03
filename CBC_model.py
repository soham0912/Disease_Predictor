import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Symptoms List
symp_list = pd.read_csv("Training.csv", nrows=0).columns.tolist()
symp_list.pop()

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
y = df['prognosis']
y = le.fit_transform(y)
y = np.ravel(y)


# Display the mapping of categories to numerical labels
category_mapping = {category: label for label, category in enumerate(le.classes_)}
print(category_mapping)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def CBC():
    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Get the indices of categorical features
    categorical_features_indices = [X_train.columns.get_loc(feature) for feature in categorical_features]

    # Define CatBoost classifier
    catboost_classifier = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass')


    # Train the classifier
    catboost_classifier.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categorical_features_indices, early_stopping_rounds=10, verbose=True)

    # Make predictions
    y_pred = catboost_classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=True))

    # Save the model to a file
    catboost_classifier.save_model('catboost_model.bin')


CBC()
