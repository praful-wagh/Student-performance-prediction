# Student Math Score Predictor

This project is a machine learning model that predicts a student's math score based on various input variables. The model was trained on a dataset that includes data on the student's gender, race/ethnicity, parental level of education, lunch status, test preparation course completion, and scores in reading, writing, and math.


# Dataset

The dataset used to train the model was obtained from kaggle. It includes data on 1000 students and 7 features.


# Model

The model was developed using the scikit-learn library in Python. It uses Random-Forest algorithm with hyper tuning. The model achieved an accuracy score of 83%.


# Usage

To use the model, you can input the student's information in the following format and press "Predict your maths score button". The model will then output a predicted math score for the student.

### Student Marks Prediction<br>
Gender : Male<br>
Race or Ethnicity : Group B<br>
Parental Level of Education : Bachelor's degree<br>
Lunch Type : Standard<br>
Test preparation Course : Completed<br>
Writing Score out of 100 : 78<br>
Reading Score out of 100 : 89<br>

The prediction is 85.75


# Limitations

It's important to note that the model's predictions are based on the input variables used in the dataset. Other factors that may affect a student's math score, such as their individual learning style or outside factors like family or personal issues, are not accounted for in the model. As such, the model should be used as a tool to assist in predicting a student's math score, but should not be relied upon as the sole determinant of a student's academic performance.\


# Future Work

Possible areas for future work on this project could include exploring additional input variables that may affect a student's math score, as well as fine-tuning the model to achieve even higher accuracy scores.
