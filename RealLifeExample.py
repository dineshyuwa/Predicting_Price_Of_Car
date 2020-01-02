import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("1.04. Real-life example.csv")
print(data.head())
print(data.describe(include="all"))
data=data.drop(["Model"],axis=1)#since more unique categorical variable in model

print("")
print(data.describe(include="all"))
print(data.isnull().sum())

data=data.dropna(axis=0)#deleting all the null variable rows

print(data.describe(include="all"))

sns.distplot(data["Price"])
print(plt.show())

q=data["Price"].quantile(0.99)
data=data[data["Price"]<q]#removing the outliers in price

# sns.distplot(data["Price"])
# print(plt.show())

# sns.distplot(data["Year"])
# print(plt.show())


q1=data["Year"].quantile(0.01)
data=data[data["Year"]>q1]#removing the out,iers in year
print(data.describe())


# sns.distplot(data["Mileage"])
# print(plt.show())


q2=data["Mileage"].quantile(0.99)
data=data[data["Mileage"]<q2]#removing the outliers in mileage
print(data.describe())

# sns.distplot(data["EngineV"])
# print(plt.show())

data=data[data["EngineV"]<6.5]#removinh the engine value which is not suitable
# sns.distplot(data["EngineV"])
# print(plt.show())

data=data.reset_index(drop=True)
print(data.describe(include="all"))

f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data['Year'],data['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data['EngineV'],data['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data['Mileage'],data['Price'])
ax3.set_title('Price and Mileage')

print(plt.show())#not normal distribution

log_price=np.log(data["Price"])
data["logPrice"]=log_price
print(data.describe())

f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data['Year'],data['logPrice'])
ax1.set_title('Log_Price and Year')
ax2.scatter(data['EngineV'],data['logPrice'])
ax2.set_title('Log_Price and EngineV')
ax3.scatter(data['Mileage'],data['logPrice'])
ax3.set_title('Log_Price and Mileage')
print(plt.show())

data=data.drop(["Price"],axis=1)
print(data.describe())

variables=data[["Mileage","Year","EngineV"]]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(variables.values,i) for i in range (variables.shape[1])]
vif["features"]=variables.columns
print(vif)

data=data.drop(["Year"],axis=1)

data_with_dummies=pd.get_dummies(data=data,drop_first=True)
print(data_with_dummies.head())
print(data_with_dummies.columns.values)
cols=['logPrice','Mileage','EngineV', 'Brand_BMW','Brand_Mercedes-Benz',
 'Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen',
 'Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van',
 'Engine Type_Gas','Engine Type_Other','Engine Type_Petrol',
 'Registration_yes']

data_pre_processed=data_with_dummies[cols]
print(data_pre_processed.head())

target=data_pre_processed["logPrice"]
inputs=data_pre_processed.drop(["logPrice"],axis=1)

scaler=StandardScaler()
scaler.fit(inputs)
scaled_inputs=scaler.transform(inputs)
x_train,x_test,y_train,y_test=train_test_split(scaled_inputs,target,test_size=0.2,random_state=365)


regression=LinearRegression()
regression.fit(x_train,y_train)
y_hat=regression.predict(x_train)

plt.scatter(y_train,y_hat)
plt.xlabel("y_train")
plt.ylabel("y_hat")
plt.xlim(6,13)
plt.ylim(6,13)
print(plt.show())

sns.distplot(y_train-y_hat)
print(plt.show())

r2=regression.score(x_train,y_train)
print(r2)

coefficient=regression.coef_
intercept=regression.intercept_

reg_summary=pd.DataFrame(inputs.columns.values,columns=["Features"])
reg_summary["Weights"]=coefficient
print(reg_summary)

y_test=y_test.reset_index(drop=True)
y_hat_test=regression.predict(x_test)
prediction_comparison=pd.DataFrame(np.exp(y_hat_test),columns=["Predictions"])
prediction_comparison["Targets"]=np.exp(y_test)
prediction_comparison["Residuals"]=prediction_comparison["Predictions"]-prediction_comparison["Targets"]
prediction_comparison["Difference%"]=np.abs(prediction_comparison["Residuals"]/prediction_comparison["Targets"])*100

print(prediction_comparison)








