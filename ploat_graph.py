import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style


# ------ collecting data from file ---------
data = pd.read_csv('student_data.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]


# ------ plot data in graph -------
p = 'G1'
style.use('ggplot')  # graph style
pyplot.scatter(data[p], data['G3'])
"""
plot points (x, y) where x is values of column G1
and y is values of column G3
"""
pyplot.xlabel(p)  # text displayed on x-axis
pyplot.ylabel("Final Grade")  # text displayed on y-axis
pyplot.show()  # display graph

