from flask import Flask, request
from JSON_Converter import JSON_to_Image
import mediapipe as mp 
from Detect_Hand_Signs_Live import Detect_Hands, Retrieve_Landmark_Data, Count_Fingers_Up, Hand_Sign_Detection

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    
    # Initialize mediapipe models
    mp_hands = mp.solutions.hands
    
    json_image = request.json
    
    queried_image = JSON_to_Image(json_image) 
    
    with mp_hands.Hands(min_detection_confidence=0.8, max_num_hands = 1) as hands:
        
        image, results = Detect_Hands(queried_image, hands)
        
        landmark_data = Retrieve_Landmark_Data(image, results)
        
        finger_count = Count_Fingers_Up(landmark_data)
        
        return Hand_Sign_Detection(finger_count)
    
if __name__ == '__main__':
    
    app.run(debug=True)
                