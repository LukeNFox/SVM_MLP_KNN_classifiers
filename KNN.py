import pandas as pd 
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
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

# define model to be used for cross validation
model_crossval = KNeighborsClassifier(n_neighbors=11)
# implement cross validation and print result
scores = cross_val_score(model_crossval , x, y, cv=10)
mean = scores.mean()
print(scores, " Mean value = ", mean)
    


##practice 

#samples a training set while holding out 10% for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#train the model using training data
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

#test the trained models accuracy using data reserved for testing
acc = model.score(x_test, y_test)
print(acc)

#predict outcome of x test data
predicted = model.predict(x_test)
names = ["c_avellana","c_americana","c_cornuta"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]],  "Actual: ", names[y_test[x]])

