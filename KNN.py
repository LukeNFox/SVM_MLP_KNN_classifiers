import pandas as pd 
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing

data = pd.read_csv("hazelnuts.csv")

predict = "variety"

le = preprocessing.LabelEncoder()
variety = le.fit_transform(list(data[predict]))

x = np.array(data.drop([predict], 1))
y = list(variety)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["c_avellana","c_americana","c_cornuta"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]],  "Actual: ", names[y_test[x]])
