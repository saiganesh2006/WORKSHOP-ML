# SA ASSIGNMENT-1
## Developed By : D.B.V. SAI GANESH
## Register Number : 212223240024
## Dept : AIML
## Objective 1 :
To Create a scatter plot between cylinder vs Co2Emission (green color)
## Code :
```
Developed by : D.B.V.SAI GANESH
Register Number : 212223240024

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fuelconsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
## OUTPUT:
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/22aa1619-b951-44f7-b866-cc128a4416a0)

## Objective 2 :
Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
Code :
```
Developed by : D.B.V.SAI GANESH
Register Number : 212223240024
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fuelconsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
## Output :
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/fa625f60-59b1-4e7b-b3ea-8b0ec8db3e7a)

## Objective 3 :
Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
## Code :
```
Developed by : D.B.V.SAI GANESH
Register Number : 212223240024

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fuelconsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='orange', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
## Output :
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/6ad08353-beb5-4f11-a741-f9a70441462a)

## Objective 4 :
Train your model with independent variable as cylinder and dependent variable as Co2Emission
## Code :
```

Developed by : D.B.V.SAI GANESH
Register Number : 212223240024

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('fuelconsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)
```
## Output :
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/b8c4c969-b335-4cf5-afde-5ca44b9367b1)

## Objective 5 :
Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission
Code :
```
Developed by : D.B.V.SAI GANESH
Register Number : 212223240024

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('fuelconsumption.csv')

X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
```
Output :
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/8c0e346b-ff8c-4590-ac41-903d2be4735c)

## Objective 6 :
Train your model on different train test ratio and train the models and note down their accuracies
## Code :
```
Developed by : D.B.V.SAI GANESH
Register Number : 212223240024

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.1, 0.4, 0.5, 0.8]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
```
## Output :
![image](https://github.com/saiganesh2006/WORKSHOP-ML/assets/145742342/9c53c4b3-969c-4c60-b309-8f498cac1895)


## Result :
All the programs executed successfully
