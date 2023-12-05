import mediapipe as mp
import cv2
import numpy as np 
import os
import math as m
import joblib
from Detect_Hand_Signs_Live import Detect_Hands, Draw_Hand_Landmarks, Retrieve_Landmark_Data, Detect_Waving, Display_Bounding_Box

def main():
    
    # Set up mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # Number to handsign
    sign_dic = {
    0: 'Open Hand',
    1: 'Peace',
    2: 'Surf Hand',
    3: 'I Love You'
    }   
    
    # Initialize prevous_landmark_data
    prev_landmark_data = []

    # Set up live webcam feed
    webcam = cv2.VideoCapture(0)    

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1) as hands: 
        while webcam.isOpened():
            ret, frame = webcam.read()

            if not ret:
                print('Camera not available')
                break
                        
            image, results = Detect_Hands(frame, hands)
            
            cv2.rectangle(image, (0,0), (640, 40), (18, 28, 179), -1)
            
            if results.multi_hand_landmarks:
                
                # Render results and draw landmark
                Draw_Hand_Landmarks(image, results, mp_hands, mp_drawing)
                
                # Proces data for model input                
                landmark_data = Retrieve_Landmark_Data(image, results)                
                
                image_data = []

                for i in range(len(landmark_data)):
                    x_coord = landmark_data[i][1]
                    y_coord = landmark_data[i][2]
                        
                    image_data.append(x_coord)
                    image_data.append(y_coord)

                pred_raw = class_model.predict([np.array(image_data)])
                hand_sign = sign_dic[int(pred_raw[0][0])]
                
                Display_Bounding_Box(image, landmark_data, hand_sign)
                
                # Detect waving and display
                prev_landmark_data = Detect_Waving(image, landmark_data, prev_landmark_data, hand_sign)

            cv2.imshow('Live Hand Tracking', image)

            # Escape mechanism when q is pressed 
            if cv2.waitKey(1) == ord('q'):
                break
            
    # Properly close everything    
    webcam.release()   
    cv2.destroyAllWindows()
    
if __name__ == '__main__':

    class_model = joblib.load('Hand_Sign_Detection/hand_sign_model.pkl')
    
    main()