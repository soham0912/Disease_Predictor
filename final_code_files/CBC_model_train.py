import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the data
data = pd.read_csv("Final_Dataset.csv")

# Split data into features and target
X = data.drop("prognosis", axis=1)
y = data['prognosis']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Display the mapping of categories to numerical labels
category_mapping = {category: label for label, category in enumerate(le.classes_)}
print(category_mapping)


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
