## 1. Approach

    -> Load Data
    -> Data Augmentation
    -> Model Building (CNN)
    -> Model Training
    -> Evaluation
    -> Save Model


## 2. Methodology

  ### * Image Data Preprocessing :
     -> Loaded chest X-ray images from dataset.
     -> All images resized to a 150x150 to maintain uniform input shape for CNN and reduce training time.
     -> Pixel values normalized to range 0–1 to improve training stability.
     -> Data augmentation (rotation, zoom, shift, flip) was applied to reduce overfitting and improve generalization.


  ### * Model Architecture (CNN) :
     -> A Convolutional Neural Network (CNN) was used for feature extraction and classification.
     -> Convolutional layers were used to detect edges, textures, and patterns in X-ray images.
     -> MaxPooling layers reduced spatial dimensions and helped prevent overfitting.
     -> Fully connected Dense layers were used for final classification.
     -> Dropout layers were used to reduce overfitting.
     -> Final output layer used sigmoid activation for binary classification.

  ### * Training Strategy :
     -> Binary Crossentropy loss function was used for classification.
     -> Adam optimizer was used for efficient gradient descent.
     -> Model was trained for multiple epochs with validation monitoring.
     -> Early stopping was used to avoid overfitting by stopping training when validation performance stopped improving.
     -> ReduceLROnPlateau was used to automatically reduce learning rate when validation loss stopped improving.

     
  ### * Threshold Tuning & Evaluation :
     -> Instead of using default 0.5 threshold, optimal threshold was calculated using validation set.
     -> Best threshold was selected based on highest F1-score.
     -> Model performance was evaluated using Confusion Matrix, Precision, Recall, and F1-score.
     -> Focus was given to recall for PNEUMONIA class, since false negatives are critical in medical diagnosis.

  ### * Model Saving :
     -> Final trained model was saved after achieving best validation performance.


## 3. Findings

  ### * Confusion Matrix :
                         [191  43]
                         [ 18 372]

     -> The confusion matrix shows that the model correctly classified 191 NORMAL cases and 372 PNEUMONIA cases.
     -> 43 NORMAL cases were incorrectly classified as PNEUMONIA.
     -> And 18 PNEUMONIA cases were missed.
     -> Overall, the model shows strong detection capability with low missed pneumonia cases, which is important for medical safety.

  ### * Classification Report :
     -> Class NORMAL = Precision: 0.91 , Recall: 0.82 , F1-score: 0.86
     -> Class PNEUMONIA = Precision: 0.90 , Recall: 0.95 , F1-score: 0.92
     -> Overall Accuracy = 90%

     -> The model performs very well on both classes, with particularly strong recall for pneumonia detection.
     -> High recall (0.95) ensures most pneumonia cases are correctly identified, which is critical in medical applications.
     -> The balance between precision and recall shows the model is stable and not heavily biased toward one class.
     
  ### * ROC-AUC Score :
     -> ROC-AUC was calculated to evaluate model performance across all classification thresholds.
     -> A higher AUC value indicates better separability between the two classes.
     -> In this model, an ROC-AUC score of 0.95 indicates very strong classification performance.

  ### * Final Report :
     -> The CNN model achieved 90% accuracy with strong performance in pneumonia detection.
     -> The model shows high recall for pneumonia cases, making it effective at identifying most positive cases.
     -> Data augmentation helped improve generalization and reduce overfitting.
