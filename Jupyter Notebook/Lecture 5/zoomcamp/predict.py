# ### Load the model
# customer = {
#  'gender': 'female',
#  'seniorcitizen': 0,
#  'partner': 'yes',
#  'dependents': 'no',
#  'tenure': 1,
#  'phoneservice': 'no',
#  'multiplelines': 'no_phone_service',
#  'internetservice': 'dsl',
#  'onlinesecurity': 'no',
#  'onlinebackup': 'yes',
#  'deviceprotection': 'no',
#  'techsupport': 'no',
#  'streamingtv': 'no',
#  'streamingmovies': 'no',
#  'contract': 'month-to-month',
#  'paperlessbilling': 'yes',
#  'paymentmethod': 'electronic_check',
#  'monthlycharges': 29.85,
#  'totalcharges': 29.85,
# }

from flask import Flask, request, jsonify
import pickle

app = Flask('churn')

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    model, fdv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    # parse the json being passed to turn it into a python dictionary
    customer = request.get_json()

    input = fdv.transform([customer])
    prediction = model.predict_proba(input)[0, 1]

    result = {
        'churn_probability': prediction,
        'churn': 'Yes' if prediction >= 0.5 else 'No'
    }
    # print(result)
    return jsonify(result)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=9696)
# print(f'Customer: {customer}')
# print(f'Churn: ', 'Yes' if output >= 0.5 else 'No')