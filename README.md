# RCA-LF
Dense Light Field Reconstruction Using Residual Channel Attention Networks

The trained model can be downloaded at: https://drive.google.com/file/d/1TMEztyrmBNSj5CePtJLi5GZrDM0gOj2s/view?usp=sharing

For training: 1- Download training data from: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/ and save the training data at ./data/TrainingData/Training/{Dataset Folder}

2- Create these directories: a) checkpoints: for saving the trained weights b) data/output/: to save the reconstructed LFs c) data/train: for saving the training patches d) data/testLF/: for saving the testing patches

Four folders are provided for different reconstruction tasks:

3- run aug_train_RCA_LF.m (Matlab) to generate training patches

4- run RCA_LF.py to train the model from scratch where MODEL>py is the model file

For testing: 1- Download testing data from: a) 30 Scenes dataset: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/ b) Reflective and occlusion datasets: http://lightfields.stanford.edu/LF2016.html and save the test data at ./data/TrainingData/Test/{Dataset Folder}

2- run aug_test_RCA_LF.m (Matlab) to generate testing patches

3- Then, run test_and_save.py to test the model and save the reconstructed LFs
