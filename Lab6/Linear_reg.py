import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#creating the table

d = {'Ages': [17.5,22,29.5,44.5,64.5,80], 'number of driver deaths': [38,36,24,20,18,28]}
df = pd.DataFrame(d)
print(df)

cols = ['Ages','number of driver deaths']
sns.pairplot(df[cols],size=2.5)
plt.tight_layout()
plt.show()

import numpy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


x = df [['Ages']].values
y= df [['number of driver deaths']].values



# now creating the linear regression object
lin_Regressor = LinearRegression()

#fitting the model to our data
lin_Regressor.fit(x, y)



plt.scatter(x, y, color = 'red')
plt.plot(x, lin_Regressor.predict(x), color = 'blue')
plt.title('Scatter plot')
plt.xlabel('Ages')
plt.ylabel('number of driver deaths')
plt.show()

#calculating slope
print('slope: %.3f' %lin_Regressor.coef_[0])

#calculating the intercept 
print('Intercept: %.3f' %lin_Regressor.intercept_)


#calculating the line equation
val = lin_Regressor.intercept_

print("Equation y="+" "+str(round((lin_Regressor.intercept_[0]),2)) + " "+ str(round((lin_Regressor.coef_[0][0]),2))+" * x")

#Predicting values for ages 40 and 60
pred_val =[40,60]

data = pd.DataFrame(pred_val)
#Defining the predicted values for ages 40 and 60.
yPrediction = lin_Regressor.predict(data)
print("The predicted values are :")
print(yPrediction)

#calculating Pearson correlation coefficient
from scipy import stats
pearson_coef, p_value = stats.pearsonr(x,y)
print("Pearson coefficient is : ")
print(pearson_coef[0])


