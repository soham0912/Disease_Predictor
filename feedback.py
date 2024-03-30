# Importing Libraries
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# API Endpoint
@app.route('/feedback', methods=['POST'])

def fetch_data():
    # Error messages
    if 'predictions' not in request.json:
        return jsonify({'error': 'No data collected'}), 400
    if 'percentages' not in request.json:
        return jsonify({'error': 'No data collected'}), 400
    if 'symptoms' not in request.json:
        return jsonify({'error': 'No data collected'}), 400

    # Retrieving data from webpage
    percentage_data = request.json['percentages']
    symptoms = request.json['symptoms']
    symptom_names = list(symptoms.values())
    percentages_list = [[disease, percentage] for disease, percentage in percentage_data.items()]

    # Extracting the higher disease possibility
    higher_chance_disease = percentages_list[0][0]
    for i in range (0, len(percentages_list) - 1):
        if percentages_list[i][1] <= percentages_list[i+1][1]:
            higher_chance_disease = percentages_list[i+1][0]


    # Reading the CSV file and entering binary values for flagging the symptoms and accordingly appending it into the list
    csv_file_path = "Training - Copy.csv"
    headers = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
    total = 0
    li = []
    for i in range(0, len(headers) - 1):
        flag = "No"
        for j in range(0, len(symptom_names)):
            if symptom_names[j] == headers[i]:
                li.append(1)
                flag = "Yes"
        if flag == "No":
            li.append(0)

    # Writing values to the CSV file
    li.append(higher_chance_disease)
    df = pd.DataFrame([li])
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
    return jsonify({
        'symptom1': symptom_names[0],
        'symptom2': symptom_names[1],
        'symptom3': symptom_names[2]
    })


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 3002
    app.run(host=host, port=port, debug=True)