import pickle
from flask import Flask, request, jsonify
# use curl/wget to download with the links
# curl -o model1.bin https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework/model1.bin
# curl -o dv.bin https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework/dv.bin
# filenames
model_file = 'model2.bin'
dv_file = 'dv.bin'

app = Flask('subscribe')

# get model
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# get dictvectorizer
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

client = {"job": "management", "duration": 400, "poutcome": "success"}

@app.route('/predict', methods=['POST'])
def predict():
  user = request.get_json()
  X = dv.transform([user])
  score = model.predict_proba(X)[0, 1]

  result = {
     'score': score
  }

  return jsonify(result)