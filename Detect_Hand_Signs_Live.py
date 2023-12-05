# Import packages
import cv2
import mediapipe as mp
import numpy as np

def Detect_Hands(frame, model):
    
    '''
    Function to detect landmarks on a frame, given a mediapipe model.
    BGR and RGB changes to accommodate cv2 and mediapipe preferences.
    Changing the image's writeability to improve speed.   
    '''
    
    image = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB) #flip image horizontally 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def Draw_Hand_Landmarks(frame, results, hand_model, draw_model):
    
    '''
    Function to draw landmarks and connections over the detected hand by the 'Detect Hands' function on live webcam feed.
    '''
    
    for hand in results.multi_hand_landmarks:
        draw_model.draw_landmarks(frame, hand, hand_model.HAND_CONNECTIONS,
                                        draw_model.DrawingSpec(color=(64, 68, 133), thickness=2, circle_radius=4),
                                        draw_model.DrawingSpec(color=(18, 28, 179), thickness=2, circle_radius=2),)


def main():
    
    # Initialize mediapipe models
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Set up live webcam feed
    webcam = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1) as hands:
        while webcam.isOpened():
            ret, frame = webcam.read()
            
            if not ret:
                print('Camera not available')
                break
            
            # Detect hands
            image, results = Detect_Hands(frame, hands)
            
            # Display landmarks
            if results.multi_hand_landmarks:
                Draw_Hand_Landmarks(image, results, mp_hands, mp_drawing)
            
            # Display live webcam feed    
            cv2.imshow('Live Hand Tracking', image)
            
            # Escape mechanism 
            if cv2.waitKey(1) == ord('q'):
                break

    # Properly close everything    
    webcam.release()   
    cv2.destroyAllWindows()
        
    
if __name__ == '__main__':
    main()