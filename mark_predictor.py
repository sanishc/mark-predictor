import numpy as np
import pickle

print("Student Mark Prediction System\n")
g1 = int(input("Enter G1: "))
g2 = int(input("Enter G2: "))
study_time = int(input("Enter Study Time: "))
failures = int(input("Enter Failures: "))
absences = int(input("Enter Absences: "))

data = np.array([[g1, g2, study_time, failures, absences]])
pickle_in = open('student_model.pickle', 'rb')
liner = pickle.load(pickle_in)
prediction = liner.predict(data)
print("\nG3 May be ", round(prediction[0]))
