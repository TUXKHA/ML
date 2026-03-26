## 1. Approach 
    -> Load Data
    -> Data Preprocessing
    -> Feature Engineering
    -> Split Data
    -> Scaling
    -> Model and Train
    -> Evaluation
    -> Save


## 2. Methodology

  ### * Categorical Columns Conversion :
     -> Converted Output y values Yes No to 0,1 for make target usable for training,Because Logistic Regression doesnt work good with text.
     -> Grouped Rare Jobs to others ,they apper less but if didnt done then will increase columns after hot encoding.
     -> Education ranked orderly to understand the education level effect with target.
     -> Default,housing,loan converted to binary.
     -> Month converted to 1-12 , and done sin cos transformation because month 12 is closer to 1.
     -> Then done one Hot encoding to all categorical columns.
     
  ### * Numerical Columns Conversion :
     -> IQR Method used for Outlier Detection.
     -> Because of so many outliers exist and imballance dataset, i decided to do outlier capping insed of dropping outlier.
     -> Used Signed log1p for balance becouse negative balance is important.
     -> Done log1p transformation for duration,campaign and previous.
     -> previous_contact new column created so can represent pday -1 values = 0 ,means never contacted and others 1.
     -> pday -(negative) values replaced by 0 means never contacted before.
     -> After corelation checking default,poutcome_other,marital_married,day dropped.

  ### * Split , Scaling and Oversampling :
     -> created x by dropping target and y for the target.
     -> split x , y for train,test and validate (80% , 10% , 10%) . stratify used so balance 0,1 of target in train ,test ,validate data.
     -> To bring all features to the same scale, scaling done to test,val and fit scaling done to train data but only on numerical columns.
     -> Due to imbalance data , SMOTE oversampling done on training data.

  ### * Model , Train And Evaluation :
     -> Logistic Regression model used.The lbfgs solver is used for optimization because of medium size dataset, C=0.5 controls
        regularization to prevent overfitting,and max_iter=1000 ensures the model converges during training.
     -> Due The dataset imbalanced ,instead of using the default threshold (0.5),calculated The threshold that gives the maximum F1-score
        using validate data. Default threshold will not give the best balance between precision and recall.
     -> Model performance was evaluated using Confusion Matrix and Classification Report Precision, Recall, F1-score.
     -> If the results satisfies , the model is saved.


## 3. Findings :

  ### * Threshold :
     -> Best Threshold: 0.7983 , Best F1-score (validation): 0.6905
     
  ### * Confusion Matrix :
                         [3840  152]
                         [ 165  364]
     -> The confusion matrix shows that the model correctly predicted 3840 non-subscribers and 364 subscribers. However, it incorrectly
        classified 152 non-subscribers as subscribers and missed 165 actual subscribers. This indicates strong overall performance with
        slightly weaker detection of the minority class due to imbalanced dataset.
     
  ### * Classification Report :
     -> Class 0 (No) = Precision: 0.96 , Recall: 0.96 , F1-score: 0.96
     -> Class 1 (Yes) = Precision: 0.71 , Recall: 0.69 , F1-score: 0.70
     -> Overall Accuracy = 93%
     -> The model performs very well on class 0, which is the majority class.For class 1 (minority class), the model achieves a balanced
        precision (0.71) and recall (0.69), which is strong considering class imbalance.The high overall accuracy (93%) confirms good
        generalization on unseen data.
     
  ### * Final Report :
     -> The Logistic Regression model achieved 93% accuracy with an F1-score of 0.70 for the minority class. After optimizing the decision
        threshold to 0.798, the model improved its balance between precision and recall, making it more effective for imbalanced
        classification.Feature engineering, SMOTE, and threshold tuning collectively improved model performance.
        The final model is suitable for real-world deployment scenarios where class imbalance exists.

  
