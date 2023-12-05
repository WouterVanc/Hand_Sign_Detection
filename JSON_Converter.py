import json
import base64
import numpy as np 
from io import BytesIO
from PIL import Image
import cv2 

def Image_to_JSON(image_path):
    with open(image_path, 'rb') as image:
        binary_image = image.read()
        
        base64_image = base64.b64encode(binary_image).decode('utf-8')
        
        json_image = {
            'data': base64_image,
            'format': 'jpg'
        }
        
        return json_image

def JSON_to_Image(json_image_path):
    with open(json_image_path, 'r') as json_image:
        json_image_data = json.load(json_image)
        
        binary_image = base64.b64decode(json_image_data['data'])

        image_buffer = BytesIO(binary_image) 
        
        image = np.array(Image.open(image_buffer))
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb   

if __name__ == '__main__':
    
    path = 'Hand_Sign_Detection/Dataset_Benchmark/Open Hand_1.jpg'
    
    json_img = Image_to_JSON(path)
    
    with open('Hand_Sign_Detection/handsign_json_test', 'w') as output:
        json.dump(json_img, output)
        
    image = JSON_to_Image('Hand_Sign_Detection/handsign_json_test')
    
    cv2.imshow('json_to_image_test', image)
    cv2.waitKey(3000)      
        

