# Import packages
import os 
import cv2
import mediapipe as mp 
from Detect_Hand_Signs_Live import Detect_Hands, Retrieve_Landmark_Data, Count_Fingers_Up, Hand_Sign_Detection
import time
from sklearn.metrics import accuracy_score
import joblib
import numpy as np 

def Logic_Model():
    # Initialize mediapipe models
    mp_hands = mp.solutions.hands

    # Import pictures
    data_path = 'Hand_Sign_Detection/Dataset_Benchmark'
    pictures = os.listdir(data_path)
    
    start = time.time()
    
    # read in pictures and overlay with mediapipe detection to consequently predict hand signs
    with mp_hands.Hands(min_detection_confidence=0.8, max_num_hands = 1) as hands:
        
        labels = []
        predicted_labels = []
        
        for picture in pictures:
            
            # Store true labels
            labels.append(picture.split('_')[0])
            
            # Store predicted labels
            frame = cv2.imread(os.path.join(data_path, picture))
            
            image, results = Detect_Hands(frame, hands)
            
            landmark_data = Retrieve_Landmark_Data(image, results)
            
            finger_count = Count_Fingers_Up(landmark_data)
            
            hand_sign = Hand_Sign_Detection(finger_count) 
            
            predicted_labels.append(hand_sign)
            
    stop = time.time()
    
    print(f"The Accuracy of the logic model = {accuracy_score(labels, predicted_labels) * 100:.0f}%")
    print(f"The logic model's runtime = {stop-start} seconds")
    
def Classifier_Model():
    
    # Number to handsign
    sign_dic = {
    0: 'Open Hand',
    1: 'Peace',
    2: 'Surf Hand',
    3: 'I Love You'
    }   
    
    # Load classifier model 
    model = joblib.load('Hand_Sign_Detection/hand_sign_model.pkl')
    
    # Initialize mediapipe models
    mp_hands = mp.solutions.hands

    # Import pictures
    data_path = 'Hand_Sign_Detection/Dataset_Benchmark'
    pictures = os.listdir(data_path)
    
    start = time.time()
    
    # read in pictures and overlay with mediapipe detection to consequently predict hand signs
    with mp_hands.Hands(min_detection_confidence=0.8, max_num_hands = 1) as hands:
        
        labels = []
        predicted_labels = []
        
        for picture in pictures:
            
            # Store true labels
            labels.append(picture.split('_')[0])
            
            # Store predicted labels
            frame = cv2.imread(os.path.join(data_path, picture))
            
            image, results = Detect_Hands(frame, hands)
            
            landmark_data = Retrieve_Landmark_Data(image, results)
            
            image_data = []

            for i in range(len(landmark_data)):
                x_coord = landmark_data[i][1]
                y_coord = landmark_data[i][2]
                    
                image_data.append(x_coord)
                image_data.append(y_coord)

            pred_raw = model.predict([np.array(image_data)])
            hand_sign = sign_dic[int(pred_raw[0][0])]            
            
            predicted_labels.append(hand_sign)
            
    stop = time.time()
    
    print(f"The Accuracy of the classifier model = {accuracy_score(labels, predicted_labels) * 100:.0f}%")
    print(f"The classifier model's runtime = {stop-start} seconds")
    
                           
if __name__ == '__main__':
    
    Logic_Model()
    
    Classifier_Model()