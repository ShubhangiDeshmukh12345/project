import pickle
import json
import random

import config
import numpy as np


class LungCancer:

    def load_model(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH, 'r') as f:
            self.json_data = json.load(f)

    def get_pred(self,Age, Gender, Air_Pollution, Alcohol_use,
       Dust_Allergy, OccuPational_Hazards, Genetic_Risk,
       chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking,
       Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue,
       Weight_Loss, Shortness_of_Breath, Wheezing,
       Swallowing_Difficulty, Clubbing_of_Finger_Nails, Frequent_Cold,
       Dry_Cough, Snoring):
        self.Age = Age
        self.Gender = Gender
        self.Air_Pollution = Air_Pollution
        self.Alcohol_use = Alcohol_use
        self.Dust_Allergy = Dust_Allergy
        self.OccuPational_Hazards = OccuPational_Hazards
        self.Genetic_Risk = Genetic_Risk
        self.chronic_Lung_Disease = chronic_Lung_Disease
        self.Balanced_Diet = Balanced_Diet
        self.Obesity = Obesity
        self.Smoking = Smoking
        self.Passive_Smoker = Passive_Smoker
        self.Chest_Pain = Chest_Pain
        self.Coughing_of_Blood = Coughing_of_Blood
        self.Fatigue = Fatigue
        self.Weight_Loss = Weight_Loss
        self.Shortness_of_Breath = Shortness_of_Breath
        self.Wheezing = Wheezing
        self.Swallowing_Difficulty = Swallowing_Difficulty
        self.Clubbing_of_Finger_Nails = Clubbing_of_Finger_Nails
        self.Frequent_Cold = Frequent_Cold
        self.Dry_Cough = Dry_Cough
        self.Snoring = Snoring
        self.load_model()
        test_array = np.ones(len(self.json_data['columns']))
        test_array[0] = self.Age
        test_array[1] = self.Gender
        test_array[2] = self.Air_Pollution
        test_array[3] = self.Alcohol_use
        test_array[4] = self.Dust_Allergy
        test_array[5] = self.OccuPational_Hazards
        test_array[6] = self.Genetic_Risk
        test_array[7] = self.chronic_Lung_Disease
        test_array[8] = self.Balanced_Diet
        test_array[9] = self.Obesity
        test_array[10] = self.Smoking
        test_array[11] = self.Passive_Smoker
        test_array[12] = self.Chest_Pain
        test_array[13] = self.Coughing_of_Blood
        test_array[14] = self.Fatigue
        test_array[15] = self.Weight_Loss
        test_array[16] = self.Shortness_of_Breath
        test_array[17] = self.Wheezing
        test_array[18] = self.Swallowing_Difficulty
        test_array[19] = self.Clubbing_of_Finger_Nails
        test_array[20] = self.Frequent_Cold
        test_array[21] = self.Dry_Cough
        test_array[22] = self.Snoring


        # test_array = np.array([10,2,1,5,3,5,7,9,2,1,9,9,1,10,1,1,9,7,7,1,1,9,1])
        print('Test array >>', test_array)
        Result = self.model.predict([test_array])
        print('PRediction', Result)
        return Result


if __name__ == "__main__":
    res = LungCancer()
    res.get_pred()
