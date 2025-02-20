This Repository contains both the training and evaluation scripts used on the F1Tenth implementation for the paper Perception-based Quantitative Runtime Verification for Learning-enabled Cyber-Physical Systems

The directory paper_set_august_2023 contains around 1000 images from our training data for our F1Tenth Platform.
These samples were collected used the actual F1Tenth Vehicles, VICON Cameras, and the data collection scripts in the data_collection_scripts folder. 

To train pose detection:
    cd pose_detection && python3 train.py

To get the pose detection head and body timings in Table 3 run:
    cd pose_detection && python3 table_3_test_timing.py

To get the yolo bounding box timings in Table 3 run:
    cd object_detection && python3 table_3_test_timing.py

Before running the evaluation scripts, download the models from the following dropbox and place them into the corresponding saved_models folder in each directory to ensure that the models are properly loaded.  

https://www.dropbox.com/scl/fo/kqn1flcei9jmgardlpg65/AGUYHXuho2Bg8wPR-IGvDSU?rlkey=zwd3zutdkq7d3j57460zuzrym&e=1&dl=0

Results from Table 2 are output from the training script train.py of the pose detection network. The results from our training are provided in table_2_results.txt. Epoch 165 is used for our results.

Please see the attached docker file for any issues related to installation, and make sure that all PyTorch models are downloaded to the saved_models folder.
