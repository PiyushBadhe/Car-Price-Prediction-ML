

# Car Price Prediction ML 🚗

![machine-learning-linear-regression](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/machine-learning-linear-regression.svg)

![Logo](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/Miscellaneous/ML%20classification.png?raw=true)

## Need of Car Price Prediction in the first place
Used Cars so called Second-hand's car have a huge market base. Many consider to buy an Used Car intsead of buying a new one, as it's is feasible and a better investment.

The main reason for this huge market is that when one buys a New Car and sale it just another day without any default on it, the price of car reduces by 30%.

There are too many frauds in the market who not only sale wrong but also they could mislead to very wrong price of a vehicle.

To overcome this frauds and misleading ourselves from fake and improper prices, here I used this *Algorithm* predicting car values besed on some of the main features defining the values of cars by using real-world *#CarDekho* dataset to Predict the price of any used car.

## Project Description 🚀
[![Generic badge](https://img.shields.io/badge/DATA%20SCIENCE-Beginners-brightgreen)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML) [![Generic badge](https://img.shields.io/badge/LANGUAGE-PYTHON-orange)](https://github.com/PiyushBadhe/Car-Price-Prediction-ML)

Car Price Prediction is a really an interesting **Machine Learning** problem for a *beginner* as there are many factors that influence the price of a car in the second-hand market. In this Project, we will be looking at a dataset based on sale/purchase of cars where our end goal will be predicting the price of the car given its features to maximize the profit.

## Dataset Required 🛠

[Car_dataset.csv](https://github.com/PiyushBadhe/Car-Price-Prediction-ML/blob/main/Car_dataset.csv)


## Modules used 🔗


| `NumPy` | `Pandas` | `MatplotLib` | `Seaborn` | `SciKit-Learn` | `Seaborn` | `Pickle` |
| :-------- | :------- | :-------- | :------- | :-------- | :------- | :-------- |

## Let's Build Model now! ⚡️⚡️


#### 1. Data Preprocessing

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

#### 2. Data Preparation

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















## Running Tests

To run tests, run the following command

```bash
  no run command detected
```


## Support 💬

**Please Feel free to support**

![Generic badge](https://img.shields.io/badge/Write-to-yellow) badhepiyuraj1997@gmail.com

[![Generic badge](https://img.shields.io/badge/CONNECT-LinkedIN-blue)](https://www.linkedin.com/in/piyush-badhe-626a9515b)

## Screenshots

![Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


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
