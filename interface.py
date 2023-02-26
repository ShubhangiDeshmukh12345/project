from flask import Flask, render_template, request
from utils import LungCancer
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


model = pickle.load(open('Naive_Bayes_Model_new.pkl', 'rb'))

app = Flask('__name__')


@app.route('/')
def homepage():
    print('Lung_HOme_page')
    return render_template('cancer_final.html')


@app.route('/predict_risk', methods=['GET', 'POST'])
def process_data():
    if request.method == 'POST':
        Age = request.form['Age']
        Gender = request.form['Gender']
        Air_Pollution = request.form['Air_Pollution']
        Alcohol_use = request.form['Alcohol_use']
        Dust_Allergy = request.form['Dust_Allergy']
        OccuPational_Hazards = request.form['OccuPational_Hazards']
        Genetic_Risk = request.form['Genetic_Risk']
        chronic_Lung_Disease = request.form['chronic_Lung_Disease']
        Balanced_Diet = request.form['Balanced_Diet']
        Obesity = request.form['Obesity']
        Smoking = request.form['Smoking']
        Passive_Smoker = request.form['Passive_Smoker']
        Chest_Pain = request.form['Chest_Pain']
        Coughing_of_Blood = request.form['Coughing_of_Blood']
        Fatigue = request.form['Fatigue']
        Weight_Loss = request.form['Weight_Loss']
        Shortness_of_Breath = request.form['Shortness_of_Breath']
        Wheezing = request.form['Wheezing']
        Swallowing_Difficulty = request.form['Swallowing_Difficulty']
        Clubbing_of_Finger_Nails = request.form['Clubbing_of_Finger_Nails']
        Frequent_Cold = request.form['Frequent_Cold']
        Dry_Cough = request.form['Dry_Cough']
        Snoring = request.form['Snoring']
        print(type(Age))
        arr = np.array([[Age, Gender, Air_Pollution, Alcohol_use,Dust_Allergy, OccuPational_Hazards, Genetic_Risk,
          chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking,
          Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue,
          Weight_Loss, Shortness_of_Breath, Wheezing,
          Swallowing_Difficulty, Clubbing_of_Finger_Nails, Frequent_Cold,
          Dry_Cough, Snoring]])
        arr = arr.astype(int)
        print(arr)
        pred = model.predict(arr)
        print(pred)
       #  res = LungCancer()
       #  Risk = res.get_pred(Age, Gender, Air_Pollution, Alcohol_use,
       # Dust_Allergy, OccuPational_Hazards, Genetic_Risk,
       # chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking,
       # Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue,
       # Weight_Loss, Shortness_of_Breath, Wheezing,
       # Swallowing_Difficulty, Clubbing_of_Finger_Nails, Frequent_Cold,
       # Dry_Cough, Snoring)
        if pred == 1:
            Res = 'LOW RISK'
        elif pred == 2:
            Res = 'MEDIUM RISK'
        elif pred == 3:
            Res = 'HIGH RISK'

        return render_template('output1.html', Risk=Res)
    else:
        return render_template('cancer_final.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=3300)
