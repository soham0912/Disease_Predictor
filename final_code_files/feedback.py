# Importing Libraries
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import subprocess
import json

app = Flask(__name__)
CORS(app)

# API Endpoint
@app.route('/feedback', methods=['POST'])

def fetch_data():
    # Error messages
    if 'predictions' not in request.json:
        return jsonify({'Predictions error': 'No data collected'}), 400
    if 'percentages' not in request.json:
        return jsonify({'Percentages error': 'No data collected'}), 400
    if 'symptoms' not in request.json:
        return jsonify({'Symptoms error': 'No data collected'}), 400

    # Retrieving data from webpage
    percentage_data = request.json['percentages']
    symptoms = request.json['symptoms']
    # symptom_names = list(symptoms.values())
    percentages_list = [[disease, percentage] for disease, percentage in percentage_data.items()]

    # Finding the maximum percentage
    max_percentage = max(percentage for _, percentage in percentages_list if percentage >= 40)

    # Collect diseases with the maximum percentage
    diseases_with_max_percentage = [disease for disease, percentage in percentages_list if percentage == max_percentage]


    # Reading the CSV file and entering binary values for flagging the symptoms and accordingly appending it into the list
    csv_file_path = "Final_Dataset.csv"
    headers = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
    li = []
    for i in range(0, len(headers) - 1):
        flag = "No"
        for j in range(0, len(symptoms)):
            if symptoms[j] == headers[i]:
                li.append(1)
                flag = "Yes"
        if flag == "No":
            li.append(0)

    lo = 1
    # Writing values to the CSV file
    for k in range(0, len(diseases_with_max_percentage)):
        if lo != 1:
            li.pop()
        li.append(diseases_with_max_percentage[k])
        df = pd.DataFrame([li])
        df.to_csv(csv_file_path, mode='a', header=False, index=False)
        lo += 1

    # Executing CBC model
    filename = "CBC_model_train.py"
    subprocess.run(["python", filename])

    # Executing ANN
    filename = "deep_learning_model_training.py"
    subprocess.run(["python", filename])

    return jsonify({
        'Message': "Run Successful"
    })



if __name__ == '__main__':
    host = '127.0.0.1'
    port = 3002
    app.run(host=host, port=port, debug=True)