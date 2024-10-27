import pickle

# ### Load the model
customer = {
 'gender': 'female',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'no',
 'tenure': 1,
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'monthlycharges': 29.85,
 'totalcharges': 29.85,
}

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    model, fdv = pickle.load(f_in)

input = fdv.transform([customer])
output = model.predict_proba(input)[:,1]

print(f'Customer: {customer}')
print(f'Churn: ', 'Yes' if output >= 0.5 else 'No')