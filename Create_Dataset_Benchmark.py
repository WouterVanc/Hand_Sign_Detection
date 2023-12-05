# Import packages
import cv2
import numpy as np 
import os 

def main():
    
    data_path = 'Hand_Sign_Detection//Dataset_Benchmark'
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
    number_of_signs = 4
    number_of_pictures = 3 # per sign
    sign_dic = {
        0: 'Open Hand',
        1: 'Peace',
        2: 'Surf Hand',
        3: 'I Love You'
    }
    
    # Initialize webcam 
    webcam = cv2.VideoCapture(0)
    
    # Iterate over hand signs to capture the images for each class
    for sign in range(number_of_signs):
        
        print(f"Prepare to capture images for {sign_dic[sign]}")
        
        while webcam.isOpened():
            
            ret, frame = webcam.read()
            
            if not ret:
                print('Camera not available')
                break
            
            cv2.putText(frame, 'Press c for capture', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Capture hand signs', frame)
            
            # Press c to quit this loop and start next loop where images are recorded
            if cv2.waitKey(1) == ord('c'):
                break
            
        count = 1 
        while count <= number_of_pictures:
            
            ret, frame = webcam.read()
            
            if not ret:
                print('Camera not available')
                break
            
            cv2.imshow('Capture hand signs', frame)    
            cv2.waitKey(1000) # /1000 = seconds between capturse
            cv2.imwrite(os.path.join(data_path, f"{sign_dic[sign]}_{count}.jpg"), frame)
            
            count += 1 
            
        print('All images have been captured') 

    # Properly close everything    
    webcam.release()   
    cv2.destroyAllWindows()
    
    # Remove excess pictures at random to have a total of 10. 
    pictures = os.listdir(data_path)
    
    excess_pictures = np.random.randint(1,13,2)     
    
    count = 1
    for picture in pictures:
        if count in excess_pictures:
            os.remove(os.path.join(data_path, picture))
            
        count += 1
            
                        
if __name__ == '__main__':
    main()            
    
    
    
    
