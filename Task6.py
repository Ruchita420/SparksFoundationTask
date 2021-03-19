# THE SPARKS FOUNDATION
# THE SPARKS FOUNDATION DATA SCIENCE AND BUSINESS ANALYTICS TASK 
# TASK 6:PREDICTION USING DECISION TREE ALGORITHM
# 
# BY-RUCHITA SIDAR



#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




#LOADING THE DATA
iris=pd.read_csv("C:/Users/Sidar/Desktop/iris.csv")
iris=pd.DataFrame(iris)
iris.head()




#CHECKING NULL VALUES
iris.isnull().sum()





#CHECKING THE SHAPE OF DATA
iris.shape




print(iris.describe())




#CLEANING DATA
iris=iris.drop('Id',axis=1)


sns.pairplot(iris)


# MODEL CREATION
from sklearn.model_selection import train_test_split
X = iris.drop('Species',axis=1)
y =iris['Species']

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.1)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)


# PREDICTION AND EVALUATION
print(X_test)
pred = classifier.predict(X_test)
pred


# VISUALIZING THE DECISION TREE
from sklearn.tree import plot_tree
f_name=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
c_name=['setosa','versicolor','verginica']
fig=plt.figure(figsize=(30,30))
plot_tree(classifier, feature_names = f_name, class_names = c_name, filled = True,rounded=True)
plt.show()


#ACCURACY
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, pred))


# USER INPUT PREDICTION
data=[float(num) for num in (input("Enter the data(Sepal.length,Sepal.width,Petal.length,Petal.width:)").strip().split(" "))]
predictions=classifier.predict([data])
print(predictions)

