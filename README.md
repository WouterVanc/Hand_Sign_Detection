# Hand_Sign_Detection

This repository contains a computer vision project to detect hand signs and waving on a live webcam feed and on saved images. 
The repository contains two models:
    - A model that detects hand signs based on the position of the hand landmarks
    - A classifier model that has been trained on a dataset to detect handsigns

You can find the necessary packages to run these script in the requirements.txt file. 

Once the right packages are installed, both the [Detect_Hand_Signs_Live.py](Detect_Hand_Signs_Live.py) (=positional logic model) 
and the [Detect_Hand_Signs_Live_Classifier.py] (=classification model) can detect four hand signs (open hand, peace, surf hand. I love you)
on a live webcam feed, as well as waving at the camera. Press 'q' to stop your session. 

The [Benchmark_Models.py] calculates the run time and accuracy of both models on 10 images.

You can create your own set of 10 images with [Create_Dataset_Benchmark.py]. 
Run the script and press 'c' to start capturing the specific hand sign once the webcam feed pops-up. 

Both [API.py] and [API_classifier.py] create an endpoint on the /predict route.
It requests an image of one of the four hand signs in JSON format and returns its prediction.

You can convert your image to a JSON object (and back) using the [JSON_converter.py]

If you wish to train the classifier on other hand signs, use [Create_Dataset_Classifier.py].
Make the necessary changes to the number of hand signs and the number of images per handsign.
Run the script and press 'c' to start capturing the specific hand sign once the webcam feed pops-up.

Next, use [Train_Classifier.py] to train a classifier model on the dataset. 
Finally, the other classifier script will be adjusted accordingly. 
