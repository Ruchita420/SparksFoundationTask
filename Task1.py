'''THE SPARKS FOUNDATION 
DATA SCIENCE AND BUSINESS ANALYTICS TASKS
TASK1-- PREDICTION USING SUPERVISED ML
BY-RUCHITA SIDAR'''

#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data=pd.read_csv(url)
print("Data successfully imported!")
data.head()

#PLOTTING THE DATA TO SEE IF ANY LINEAR RELATIONSHIP EXIST
data.plot(x='Hours', y='Scores', style="o")
plt.title("Hours vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.show()

#PREPARING THE DATA FOR MODEL TRAINING
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#SPLITTING THE DATA INTO TRAINING AND TEST SET BY USING SCIKIT-LEARNS'S BUILT-IN TRAIN_TEST_SPLIT() METHOD
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#TRAINING THE ALGORITHM
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print('Training Complete')

#PLOTTING THE REGRESSION LINE AND DATA
line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.title("Fitted Model Plot")
plt.xlabel("No. of hours studied")
plt.ylabel("Score in percentage")
plt.show()

#MAKING PREDICTIONS
print(x_test)
y_pred = regressor.predict(x_test)

#COMPARING ACTUAL V/S PREDICTED
df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
print(df)

#WE CAN ALSO TEST WITH OUR OWN DATA
hrs=9.25
test=np.array([hrs])
test=test.reshape(-1,1)
pred = regressor.predict(test)
print("No of Hours={}".format(hrs))
print("Predicted Score = {}".format(pred[0]))

