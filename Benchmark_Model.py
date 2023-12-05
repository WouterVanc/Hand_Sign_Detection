# Import packages
import os 
import cv2
import mediapipe as mp 
from Detect_Hand_Signs_Live import Detect_Hands, Retrieve_Landmark_Data, Count_Fingers_Up, Hand_Sign_Detection
import time
from sklearn.metrics import accuracy_score


def main():
    # Initialize mediapipe models
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Import pictures
    data_path = 'Hand_Sign_Detection//Dataset_Benchmark'
    pictures = os.listdir(data_path)
    
    start = time.time()
    
    # read in pictures and overlay with mediapipe detection to consequently predict hand signs
    with mp_hands.Hands(min_detection_confidence=0.8) as hands:
        
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
    
    print(f"The Accuracy of the model = {accuracy_score(labels, predicted_labels) * 100:.0f}%")
    print(f"The model's runtime = {stop-start} seconds")
                           
if __name__ == '__main__':
    main()