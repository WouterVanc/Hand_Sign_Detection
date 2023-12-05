import cv2
import os

def main():
    data_path = 'Hand_Sign_Detection/Dataset_Classifier'
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
    number_of_signs = 4
    number_of_pictures = 200 # per sign
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
        if not os.path.exists(os.path.join(data_path, f"{sign_dic[sign]}")):
            os.makedirs(os.path.join(data_path, f"{sign_dic[sign]}"))
        
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
            cv2.waitKey(1) # /1000 = seconds between capturse
            cv2.imwrite(os.path.join(data_path, f"{sign_dic[sign]}", f"{count}.jpg"), frame)
            
            count += 1 
            
    # Properly close everything    
    webcam.release()   
    cv2.destroyAllWindows()
            
    print('All images have been captured')    
    
if __name__ == '__main__':
    main()    