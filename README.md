# Homesite Quote Conversion Prediction

This repository contains the code and results for predicting the probability that a customer would buy a quoted insurance plan using the Homesite Quote Conversion dataset. The project involves applying various machine learning algorithms and techniques, such as SMOTE, ensemble predictions (stacking), and hyperparameter tuning, to improve model performance.


## Dataset

The dataset used in this project is from a Kaggle competition with a prize of $20,000. It is about predicting the probability that a customer would buy a quoted insurance plan. The preprocessed datasets were obtained from the following links:

- Preprocessed Training set:
  - [RevisedHomesiteTrain1.csv](https://drive.google.com/file/d/1Sc5e_vq5FuoZ5I5xegQ2-3Y3_0_qBsUn/view?usp=sharing)
  - [NewHomesiteTrain.csv](https://drive.google.com/file/d/1zY6UJf6TnWr-dfmnTmUZyV1J6UwIE6l_/view?usp=sharing)
- Preprocessed Test set:
  - [RevisedHomesiteTest1.csv](https://drive.google.com/file/d/1vmdRJmL8AYMMDruX9b1Dv0xW3q8oGQY_/view?usp=sharing)
  - [NewHomesiteTest.csv](https://drive.google.com/file/d/1q3YJYU6MgU6rKuo6Z1TnIPsU9g7_aGxk/view?usp=sharing)

## Requirements

- Python 3.6 or higher
- Scikit-learn
- Pandas
- Numpy
- Imbalanced-learn

## Steps 

1. **Data Preprocessing:** The preprocessed datasets provided were used for this project. Additional preprocessing steps, such as feature scaling and encoding, were performed as needed for each algorithm.
2. **Classification Methods:** The following classification methods from Scikit Learn were implemented and evaluated:
   - Multilayer Perceptron
   - Support Vector Machines
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors
3. **SMOTE:** To address the class imbalance issue, SMOTE (Synthetic Minority Over-sampling Technique) and its variations were applied using different sampling ratios to improve the minority class prediction.
4. **Ensemble Predictions:** One-layer stacking was performed to combine the predictions from the various algorithms. The base classifiers used in the stacking process included decision tree, random forest, support vector machines, multilayer perceptron, and K-nearest neighbors. A meta-classifier was trained on the outputs of the base classifiers to make the final prediction.
5. **Hyperparameter Tuning:** Hyperparameter tuning was carried out on the stacked model using grid search and cross-validation to optimize the model performance. Tuning of individual models was also performed to improve their respective performances.

## Instructions

1. Clone this repository and navigate to the project folder.
2. Install the required packages (if not already installed).
3. Run the Jupyter Notebook or Python script containing the data preprocessing, classification methods, SMOTE, ensemble predictions, and hyperparameter tuning.
4. Analyze the results and prepare a report, discussing the dataset/problem, the machine learning models, and the techniques used.

## Results

The final stacked model, which combined the predictions from various base classifiers (decision tree, random forest, support vector machines, multilayer perceptron, and K-nearest neighbors), used Logistic Regression as the meta-classifier. The model was built without hyperparameter tuning and with 0.5 SMOTE sampling ratio to address class imbalance.

The performance of the stacked model was evaluated using the Kaggle submission scores. The results are as follows:

- Public Score: 0.85891
- Private Score: 0.85975

The stacked model with Logistic Regression as the meta-classifier and a 0.5 SMOTE sampling ratio demonstrated a competitive performance in comparison to the individual classifiers. The model's performance could potentially be further improved with hyperparameter tuning and by exploring other ensemble methods or stacking configurations.
