from flask import Flask, request
from JSON_Converter import JSON_to_Image
import mediapipe as mp 
from Detect_Hand_Signs_Live import Detect_Hands, Retrieve_Landmark_Data
import joblib
import numpy as np 

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    
    # Initialize mediapipe models
    mp_hands = mp.solutions.hands 
    
    sign_dic = {
    0: 'Open Hand',
    1: 'Peace',
    2: 'Surf Hand',
    3: 'I love you'
    }
    
    json_image = request.json
    
    queried_image = JSON_to_Image(json_image) 
    
    with mp_hands.Hands(min_detection_confidence=0.8, max_num_hands = 1) as hands:
        
        image, results = Detect_Hands(queried_image, hands)
        
        landmark_data = Retrieve_Landmark_Data(image, results)
        
        image_data = []
        
        for i in range(len(landmark_data)):
            x_coord = landmark_data[i][1]
            y_coord = landmark_data[i][2]
                
            image_data.append(x_coord)
            image_data.append(y_coord)

        pred_raw = model.predict([np.array(image_data)])
        hand_sign = sign_dic[int(pred_raw[0][0])]
        
        return hand_sign  

if __name__ == '__main__':
    
    model = joblib.load('Hand_Sign_Detection/hand_sign_model.pkl')
    
    app.run(debug=True)
                