# Import packages
import cv2
import mediapipe as mp
import numpy as np
import math as m 

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
    Function to draw landmarks and connections over the detected hand 
    by the 'Detect Hands' function on live webcam feed.
    '''
    
    for hand in results.multi_hand_landmarks:
        draw_model.draw_landmarks(frame, hand, hand_model.HAND_CONNECTIONS,
                                        draw_model.DrawingSpec(color=(64, 68, 133), thickness=2, circle_radius=4),
                                        draw_model.DrawingSpec(color=(18, 28, 179), thickness=2, circle_radius=2),)

def Retrieve_Landmark_Data(frame, results):
    
    '''
    Function to retrieve all the hand landmarks in a list of list for ease of acces and use.
    Multiply by frame dimensions to get real coordinates instead of normalized coordinates.
    '''
    
    landmark_list = []
    for id, landmark in enumerate(results.multi_hand_landmarks[-1].landmark):
        
        height = frame.shape[0]
        width = frame.shape[1]
        
        x_real, y_real = int(landmark.x * width), int(landmark.y * height)
        
        landmark_list.append([id, x_real, y_real])
        
    return landmark_list

def Thumb_Angle(landmark_data):
    
    '''
    Function to calculate the angle between landmark 2, 3, and 4. This to counter a surf sign with a bent thumb. 
    '''
    
    lm = landmark_data
    
    v1 = [lm[4][0] - lm[3][0], lm[4][1] - lm[3][1]]
    v2 = [lm[3][0] - lm[2][0], lm[3][1] - lm[2][1]]
    
    dot_prod = np.dot(v1,v2)
    mag_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    arccos_input = dot_prod / mag_prod
    
    if not -1 <= arccos_input <= 1: 
        return 0
    
    return np.degrees(np.arccos(dot_prod / mag_prod))    

def Count_Fingers_Up(landmark_data):
    
    '''
    Function to count how many fingers are up (straight) given a set of landmark_data of one frame. 
    Straight is defined by measuring the Euclidean distance between the wrist and tip of the finger and comparing it
    to the distance of the wrist to the finger landmarks that are closer to it. 
    '''
    
    lm = landmark_data
    fingers_up = []
    
    if abs(lm[4][1] - lm[0][1]) > abs(lm[3][1] - lm[0][1]) and Thumb_Angle(lm) < 4: # horizontal distance for thumb + angle logic to make sure it is straight. 
        fingers_up.append(0)    
    
    # index finger = 1 
    if m.dist([lm[0][1], lm[0][2]],[lm[8][1], lm[8][2]]) > m.dist([lm[0][1], lm[0][2]],[lm[7][1], lm[7][2]]):
        fingers_up.append(1) 

    # middle finger = 2
    if m.dist([lm[0][1], lm[0][2]],[lm[12][1], lm[12][2]]) > m.dist([lm[0][1], lm[0][2]],[lm[11][1], lm[11][2]]):
        fingers_up.append(2)

    # ring finger = 3 
    if m.dist([lm[0][1], lm[0][2]],[lm[16][1], lm[16][2]]) > m.dist([lm[0][1], lm[0][2]],[lm[15][1], lm[15][2]]):
        fingers_up.append(3) 

    # ring finger = 4 
    if m.dist([lm[0][1], lm[0][2]],[lm[20][1], lm[20][2]]) > m.dist([lm[0][1], lm[0][2]],[lm[19][1], lm[19][2]]):
        fingers_up.append(4)
        
    return fingers_up

def Hand_Sign_Detection(finger_count):
    
    '''
    Function that infers which hand sign is portrayed base on the fingers that are straight.
    '''
    
    if finger_count == [1,2]:
        return 'Peace'
    
    if finger_count == [0,1,2,3,4]:
        return 'Open Hand'
    
    if finger_count == [0,4]:
        return 'Surf Hand'
    
    if finger_count == [0,1,4]:
        return 'I Love You'
    
def Display_Bounding_Box(frame, landmark_data, hand_sign):
    
    '''
    Function to create bounding box that adapts to the size of the hand
    and displays the hand sign on top. 
    '''
     
    x_values = [lm[1] for lm in landmark_data]
    y_values = [lm[2] for lm in landmark_data]
    
    x_min, y_min = min(x_values) - 15, min(y_values) - 15
    x_max, y_max = max(x_values) + 15, max(y_values) + 15     

    cv2.rectangle(frame, (x_min,y_min), (x_max, y_max), (0,0,0), 2)
    cv2.putText(frame, hand_sign, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)

def Detect_Waving(frame, landmark_data, prev_landmark_data, hand_sign):
    
    '''
    Function to detect waving motion based on open hand and the vertical 
    and horizontal displacement of landmarks between consecutive frames
    '''
    
    motion = ' '

    # Key parameters
    frames_to_track = 1 
    waving_threshold = 50
    vert_threshold = 30
    
    if hand_sign == 'Open Hand':
        if len(prev_landmark_data) >= frames_to_track:
            hor_displacement_top = abs(landmark_data[12][1] - prev_landmark_data[-frames_to_track][12][1]) 
            hor_displacement_bottem = abs(landmark_data[0][1] - prev_landmark_data[-frames_to_track][0][1])
            vert_displacement_bottem = abs(landmark_data[0][2] - prev_landmark_data[-frames_to_track][0][2])

            # Waving logic 
            if hor_displacement_top > waving_threshold and hor_displacement_top > 1.2 * hor_displacement_bottem and vert_displacement_bottem < vert_threshold:
                motion = 'Waving!'
            
        prev_landmark_data.append(landmark_data)
        
        # Only keep last 5 frames
        prev_landmark_data = prev_landmark_data[-frames_to_track:]
        
        cv2.putText(frame, motion, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
   
    return prev_landmark_data                 

def main():
    
    # Initialize mediapipe models
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
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
                
                # Detect handsign logic and display
                landmark_data = Retrieve_Landmark_Data(image, results)
                
                finger_count = Count_Fingers_Up(landmark_data)
                
                hand_sign = Hand_Sign_Detection(finger_count)
                
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
    main()