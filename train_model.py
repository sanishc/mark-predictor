import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle


# collecting data from file
data = pd.read_csv('student_data.csv', sep=';')


# filter columns from pandas DataFrame
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]


# generates numpy arrays from data
predict = 'G3'
x = np.array(data.drop([predict], 1))  # with values excl column predict
"""
drop([items list], list dimension)
"""
y = np.array(data[predict])  # with values of predict column only


best_accuracy = 0
for _ in range(30):
    # split 10% of x and y dataset for testing and rest for training
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # train model with training dataset using linear regression
    liner = linear_model.LinearRegression()
    liner.fit(x_train, y_train)

    # test model for accuracy
    accuracy = liner.score(x_test, y_test)
    """
    machine predicts a result for each data in x_test based on training
    and compare that result with it's actual result ie, data in y_test 
    """
    print(accuracy)

    # save best accurate model
    if accuracy > best_accuracy:
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(liner, f)
        best_accuracy = accuracy

print("Best Training Accuracy: ", best_accuracy)

# Load Saved model
pickle_in = open('student_model.pickle', 'rb')
liner = pickle.load(pickle_in)

# print test results to user
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
each time train_test_split values are taken randomly
"""
predictions = liner.predict(x_test)  # predicts results for each data in x_test

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    """
    predictions[x] : value predicted by machine
    x_test[x] : values received as input for predicting
    y_test[x] : the actual result for user to compare with prediction
    """