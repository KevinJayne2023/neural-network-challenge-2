Employee Attrition & Department Prediction Model
This project aims to predict employee attrition and department assignment using a multi-output classification neural network. By leveraging a dataset with features related to employee demographics, performance, and organizational details, the model can simultaneously predict whether an employee is likely to leave the company (attrition) and their associated department.

Project Overview
Objective: Build a machine learning model that can predict two outputs:

Employee attrition (binary classification: Yes/No)
Employee department (multi-class classification)
Key Techniques:

Multi-output neural network model
Handling imbalanced datasets
Regularization (L2) to avoid overfitting
Dataset
The dataset includes information on employees, such as:

Demographics (age, gender, marital status)
Job role, department, and level
Performance and satisfaction scores
Other features relevant to predicting attrition and department assignment

Target Variables:
Attrition: Whether the employee has left the organization (binary: Yes/No).
Department: The department in which the employee works (multi-class: Sales, R&D, HR).

Model Architecture
The project uses a neural network model designed for multi-output classification:

Input: Shared input layer using features from the dataset.
Hidden Layers: Multiple hidden layers with dense units and activation functions (ReLU and tanh).
Output Layers:
Attrition Output: Binary classification using the sigmoid activation function.
Department Output: Multi-class classification using the softmax activation function.

Key Features and Methods
Multi-Output Classification: Simultaneously predicts both attrition and department.
Imbalanced Data Handling: Techniques such as oversampling, undersampling, and class weighting were considered to address imbalanced data (especially in attrition prediction).
Regularization: L2 regularization was applied to reduce overfitting and generalize better on unseen data.

Model Evaluation
The model’s performance is evaluated using the following metrics:

Attrition: Accuracy for the binary classification of attrition.
Department: Categorical accuracy for multi-class department classification.

Results:
The model shows strong potential for predicting employee attrition and department assignment with balanced accuracy.

Future Improvements
Imbalanced Data: Further work can be done to handle class imbalance more effectively using techniques like RandomOverSampling or SMOTE (Synthetic Minority Over-sampling Technique).
Hyperparameter Tuning: The learning rate, batch size, and number of neurons can be optimized further for better performance.
Feature Engineering: Creating additional features from the dataset, or transforming existing ones, could improve the model’s ability to make accurate predictions.
Dropout: Implementing dropout layers could enhance regularization and prevent overfitting further.

Requirements
To run the project, you'll need the following libraries:

Python 3.x
TensorFlow or Keras
Scikit-learn
Pandas
Numpy

Contact
For any questions or support, feel free to reach out to Kevin Jayne at jaynekev2023@gmail.com
