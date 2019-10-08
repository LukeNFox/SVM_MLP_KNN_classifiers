import sklearn
from sklearn import svm
import pandas as pd 
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("hazelnuts.csv")

predict = "variety"


# encode prediction varible into numbers
le = preprocessing.LabelEncoder()
variety = le.fit_transform(list(data[predict]))

#create list of data and corresponding list of variables to be predicted
x = list(np.array(data.drop([predict], 1)))
y = list(variety)

model_crossval = svm.SVC(kernel="linear")
# implement cross validation and print result
scores = cross_val_score(model_crossval , x, y, cv=10)
mean = scores.mean()
print(scores, " Mean value = ", mean)