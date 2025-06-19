This is is my thesis project and it employs a robust convolutional neural network to classify punch marks in late-medieval panel paintings, and implements explainability via Grad-CAM and XGrad-CAM, which are quantitatively evaluated using IROF and ROAD metrics.


# Get started
1. Download the PSB images from: https://drive.google.com/drive/folders/1Ouga4ms22NK-sDUkI4MoFqHuhxG2qG1i?usp=drive_link
2. From each PSD image create the crops using the crops.py file apart from _07_Traino_S_Domenico_2_. This one will be used as a test image and needs to be split into a different folder.
3. After all of the crops are generated, run class_split.py to split them according to their class.
4. Run main_cnn.py with the paths for your data. This trains and saves the model
5. Once the model is saved, the Grad-CAM_multilayer.py file can be run to generate the saliency maps (explanations) with the appropiate paths. This will generate one folder per class present in the test set, each of those folders will, in turn,
contain two folders, one for Grad-CAM and one for XGrad-CAM.
6. Finally, run the faithfulness_evaluation.py with the paths to the generated saliency maps.
