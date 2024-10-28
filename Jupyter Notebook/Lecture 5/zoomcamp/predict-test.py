#!/usr/bin/env python
# coding: utf-8
import requests
url = "http://127.0.0.1:5000/predict"

customer = {
 'gender': 'female',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'yes',
 'tenure': 3,
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'yes',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'monthlycharges': 42.85,
 'totalcharges': 29.85,
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == 'Yes':
  print('Churning')
else:
  print('Not Churning')
