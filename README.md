# Hand_Sign_Detection

This repository contains a computer vision project to detect hand signs and waving on a live webcam feed and on saved images. 

You can find the necessary packages to run these script in the requirements.txt file. 

Once the right packages are installed, run the 'Detect_Hand_Signs_Live.py' for live detection with your webcam.
Press 'q' to stop. 

The 'Benchmark_Model.py' calculates the run time and accuracy of the model on 10 images.
These 10 images can be captured using 'Create_Dataset_Benchmark.py'. 
Run the script and press c to start capturing the specific hand sign once the webcam feed pops-up. 

The 'API.py' creates an endpoint on the /predict route and takes in an image in the form of JSON.
You can convert your image to a JSON object (and back) using the 'JSON_converter.py'