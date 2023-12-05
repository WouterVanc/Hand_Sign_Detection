import os
import mediapipe as mp
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from Detect_Hand_Signs_Live import Detect_Hands, Retrieve_Landmark_Data

def main():
    # Set up mediapipe
    mp_hands = mp.solutions.hands

    data_path = 'Hand_Sign_Detection/Dataset_Classifier'

    data = []
    labels = []

    with mp_hands.Hands(min_detection_confidence=0.8) as hands: 
        for sign in os.listdir(data_path):
            for image_file in os.listdir(os.path.join(data_path, sign)):
                
                image_data = [] 
                
                frame = cv2.imread(os.path.join(data_path, sign, image_file))
                
                image, results = Detect_Hands(frame, hands)
                
                landmark_list = Retrieve_Landmark_Data(image, results)
                
                for i in range(len(landmark_list)):
                    x_coord = landmark_list[i][1]
                    y_coord = landmark_list[i][2]
                    
                    image_data.append(x_coord)
                    image_data.append(y_coord)
                
                data.append(image_data)
                labels.append(sign)
        
    # Train model
    X = np.array(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=labels, random_state=101)

    classifier_model = RandomForestClassifier().fit(X_train, y_train)

    y_pred = classifier_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    joblib.dump(classifier_model, 'Hand_Sign_Detection/hand_sign_model.pkl')
    
if __name__ == '__main__':
    main()  