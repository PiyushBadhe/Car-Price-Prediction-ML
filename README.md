

# 🚗 Car Price Prediction ML 

![machine-learning-linear-regression](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/machine-learning-linear-regression.svg)

![Logo](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/ML%20classification.png?raw=true)

## 🚀 Need of Car Price Prediction in the first place
Used Cars so called Second-hand's car have a huge market base. Many consider to buy an Used Car intsead of buying a new one, as it's is feasible and a better investment.

The main reason for this huge market is that when one buys a New Car and sale it just another day without any default on it, the price of car reduces by 30%.

There are too many frauds in the market who not only sale wrong but also they could mislead to very wrong price of a vehicle.

To overcome this frauds and misleading ourselves from fake and improper prices, here I used this *Algorithm* predicting car values besed on some of the main features defining the values of cars by using real-world *#CarDekho* dataset to Predict the price of any used car.

## 📝 Project Description 
[![Generic badge](https://img.shields.io/badge/DATA%20SCIENCE-Beginners-brightgreen)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML) [![Generic badge](https://img.shields.io/badge/LANGUAGE-PYTHON-orange)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML)

Car Price Prediction is a really an interesting **Machine Learning** problem for a *beginner* as there are many factors that influence the price of a car in the second-hand market. In this Project, we will be looking at a dataset based on sale/purchase of cars where our end goal will be predicting the price of the car given its features to maximize the profit.

## 🛠 Dataset Required 

[Car_dataset.csv](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/main/Car_dataset.csv)


## ⚙️ Librariess used 

I've used a separate ML environment where only limited but required libraries were installed


| `NumPy` | `Pandas` | `MatplotLib` | `Seaborn` | `SciKit-Learn` | `Seaborn` | `Pickle` |
| :-------- | :------- | :-------- | :------- | :-------- | :------- | :-------- |

**Use !pip command to install those libraries into your Environment**

NumPy : It is an ibnbiult Library used in Python bu sometimes unexpectedly we need to download it `!pip install numpy`

Pandas : `!pip install pandas`

MatplotLib : `!pip install matplotlib`

SciKit_Learn : `!pip install sklearn`

Seaborn : `!pip install seaborn`

Pickle : `!pip install pickle`


##  Let's Build Model now! ⚡️⚡️


### 1. DATA PREPROCESSING 🏽‍

- Import very first package for data reading for carrying out preprocessing techniques on the same

```
import pandas as pd
```

- Now, assign the data values from Dataset **Car_dataset.csv** with `read_csv`

```
df = pd.read_csv('Car_dataset.csv')
```

- Visualize and validate whether the dataset is successfully assigned to the vaariable

```
df.head()
```

![df.head()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/df.head().png)

- Check size of the Dataset

```
df.shape
```

![df.shape](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/df.shape.png)


- Choosing features uniquely defines each car's properties hence varying values can be achieved

- Features used here are `Seller_Type`, `Transmission`, `Owner`, `Fuel`

- Using these features and their unique values which directly classify/differentiate each car

```
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel'].unique())
```

![Unique Values](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/Unique%20Values.png)

- Checking if presence of `NULL` values in the dataset
- 
```
df.isnull().sum()
```

![isnull()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/isnull.png)


- Describing all calculated statistical terms aka *Sum*, *Mean*, *Standard Deviation*, *Minimum*, *Maximum* etc

```
df.describe()
```

![df.describe()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/parameters%20df.describe().png)

- Fetching Columns present before **Data Preparation**

```
df.columns
```

![df.columns](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/df.columns().png)

### 2. DATA PREPARATION 🏽‍

- Neglecting unncessary column(s) from the Dataset i.e. `Car_Name` as `Car_name` may include many and is not uniquely differentiating as a feature

```
final_dataset = df[['Year', 'Selling_Price', 'Km_Driven', 'Fuel', 'Seller_Type','Transmission', 'Owner']]
final_dataset.head()
```

![final_dataset1](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/final_dataset1.png)

- We actually can add or modify features as per references for training and testing the model
- Here, we are going to add a new feature `Car_Age` to simplify how many years a particular car is been used

- Add a `Current_Year` column to Dataset having value _2021_ in all the rows as 2021 is the current year

```
final_dataset['Current_Year'] = 2021
final_dataset.head()
```

![final_dataset2](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/final_dataset2.png)

- Getting `Car_Age` with simple logic and finally adding `Car_Age` column

```
final_dataset['Car_Age'] = final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.head()
```

![final_dataset3](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/final_dataset3.png)

- As we know how is old the car now, we can neglect both the `Year` and `Current_Year` columns now

```
final_dataset.drop(['Year'], axis = 1, inplace = True)
final_dataset.drop(['Current_Year'], axis = 1, inplace = True)
final_dataset.head()
```

![final_dataset4](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/final_dataset4.png)

- Converting encoded unicode

- In case if you don't know unicoding, let me simplify your doubt just in a minute with a small table;

| {Parameter1} | {Parameter2} | {Parameter3} | Description |
| :-------- | :-------- | :------- | :-------------------------------- |
| 1 | 0  | 0 | This will represent the value is belonging to {Parameter1} |
| 0 | 1  | 0 | This will represent the value is belonging to {Parameter2} |
| 0 | 0  | 1 | This will represent the value is belonging to {Parameter3} |
|   |    |   | TIP: But also when Parameter1==0 and Parameter2==0, It will actually represent belonging to {Parameter3} itself |

```
final_dataset = pd.get_dummies(final_dataset, drop_first = True) # First column should be deleted from "dummy variable trap"

final_dataset.head()
```

![Unicode](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/Unicode.png)


### 3. DATA VISUALIZATION 🏽‍

**Now it's time to Visualize the Data prepared till now**

- Import `Seaborn` and plot a **Pairplot** very quickly

```
import seaborn as sbs
sbs.pairplot(final_dataset)
```

![sbs.pairplot](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/sbs.pairplot.png)

- Import MatplotLib as well and plot a heatmap having correlation in between the data
- For more of the `%matplotlib inline` term refer [this](https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline#:~:text=%25matplotlib%20inline%20sets%20the%20backend,stored%20in%20the%20notebook%20document.) article.

```
import matplotlib.pyplot as plt
%matplotlib inline


# Heatmapping the data
corrmat = final_dataset.corr()
top_corr_features = corrmat.index

plt.figure(figsize = (20, 20))


# Visualize the heatmap
hmap = sbs.heatmap(final_dataset[top_corr_features].corr(), annot = True, cmap = "RdYlGn")  # Color pattern chosen here = "RdYlGn"

```

![sbs.hmap](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/sbs.hmap.png)

### 4. FEATURE ENGINEERING 🏽‍

- Let's have a look again to the Dataset prepared till now

```
final_dataset.head()
```
**DEPENDENT and INDEPENDENT Features**

- Looking at the very first column of Dataset it is the `Selling_Price` we are going to predict through our ML Model
- So in this case, we won't be needing this column for our model building
- Let's then neglect `Selling_Price` then but now by using `iloc` function

```
X = final_dataset.iloc[:,1:] 
Y = final_dataset.iloc[:,0]
```

```
X.head()
````

![X.head()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/X.head().png)

```
Y.head()
```

![Y.head()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/Y.head().png)


**FEATURE Importance**

- Let's now fit our **X** and **Y** values to the model with `ExtraTreeRegressor`
- Import `ExtraTreeRegressor` from `Seaborn`

```
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, Y)
```

- Now we can know each of the features' Importance

```
print(model.feature_importances_)
```

![print(model.feature_importances_)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/print(model.feature_importances_).png)

- Are you able to understand the feature's understand? Or Can you tell which of the features is more important than another one?

- So that's exactly where Visualization plays a very important role for drawing insights from the data we couldn't understand

- Let's say we are going to plot a Graph of Features Importance

```
feat = pd.Series(model.feature_importances_, index = X.columns)
feat.nlargest(5).plot(kind = 'barh')
plt.show()
```

![plt.show()](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/plt.show().png)


### 5. TRAINING ML MODEL 🏽‍

- Whoosh! After all of the DATA PREPARATION, we can build our model for real
- But before we'd do that, we have to split the data for training and testing our model
- Training the model what is called Building a ML Model and Testing of the trained model will output the predicted Selling Price of the car which is our end goal
- For Training and Testing we'll be using 8:2 ratio data. However it is best to use more of the present data for training purpose as it'll give very great accuracy at the end
- Rest 20% of the data will be used for testing our ML model for its accuracy

```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X,Y, test_size = 0.2)

X_train.shape  # Checking the size of the dataset used for training our ML model
```

![X_train.shape](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/X_train.shape.png)


##### 5.1 Ensembling 🔗

- The goal of [ENSEMBLE METHODS](https://scikit-learn.org/stable/modules/ensemble.html) is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator

- In this project we're using [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 

```
from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor()
```

- For *Estimation*, we'll be introducing `RandomizedSearchCV'

- Randomized search on hyper parameters: RandomizedSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. For more of `RandomizedSearchCV` refer to [this Article](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

```
from sklearn.model_selection import RandomizedSearchCV
```
```
# Hyperparameters
# RandomizedSearchCV

import numpy as np

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)] # max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

```

- Create the random grid

```

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
```

![print(random_grid)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/print(random_grid).png)















## Support 💬

**Please Feel free to support**

![Generic badge](https://img.shields.io/badge/Write-to-yellow) badhepiyuraj1997@gmail.com

[![Generic badge](https://img.shields.io/badge/CONNECT-LinkedIN-blue)](https://www.linkedin.com/in/piyush-badhe-626a9515b)



## Usage/Examples

```
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
