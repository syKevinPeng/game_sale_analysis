#### Siyuan "Kevin" Peng, Yuanzhe "Siris" Zheng, Yanlin "Jacky" Liu
![Image of Yaktocat](https://cdn.mos.cms.futurecdn.net/rLh7Dh7EKo8F6zmDtXYp8W.jpg)
# Table of Contents
1. [Introduction](#introduction)
2. [Install packages](#install-pkg)
3. [Data Dowloading](#data-download)
4. [Preprocessing](#preprocessing)<br>
    a. [Load and Clean Dataset](#load-and-clean)<br>
    b. [Data Analysis and Visualization](#data-ana-vis)
5. [Machine Learning Model](#ml-model)<br>
    a. [What and Why](#what-why)<br>
    b. [Training](#training)<br>
    c. [Result Anlysis and Demonstration](#result-and-demon)
6. [Future Application](#future-app)
7. [Reference and External Link](#ref-and-extlink)

## 1. Introduction <a name="introduction"></a>
TODO:

## 2. Install Packages <a name="install-pkg"></a>
```
pip install kaggle numpy matplotlib pandas sklearn
```
or use [environment.yml](https://github.com/syKevinPeng/game_sale_analysis/blob/main/environment.yml) to install packages in Conda environment
```
conda env update -f environment.yml
```
## 3. Data Downloading <a name="data-download"></a>


```python
import kaggle
# remember to put kaggle.json to your C:/username/.kaggle
!kaggle datasets download -d ashaheedq/video-games-sales-2019
```

    video-games-sales-2019.zip: Skipping, found more recently modified local copy (use --force to force download)
    

or directly download from kaggle webpage: [https://www.kaggle.com/ashaheedq/video-games-sales-2019](https://www.kaggle.com/ashaheedq/video-games-sales-2019)_
## 4. Preprocessing <a name="preprocessing"></a>
### 4.a. Load and Clean Data <a name="load-and-clean"></a>


```python
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 
df = pd.read_csv("vgsales-12-4-2019.csv")
additional = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-07-30/video_games.csv")
additional.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>game</th>
      <th>release_date</th>
      <th>price</th>
      <th>owners</th>
      <th>developer</th>
      <th>publisher</th>
      <th>average_playtime</th>
      <th>median_playtime</th>
      <th>metascore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Half-Life 2</td>
      <td>Nov 16, 2004</td>
      <td>9.99</td>
      <td>10,000,000 .. 20,000,000</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>110.0</td>
      <td>66.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Counter-Strike: Source</td>
      <td>Nov 1, 2004</td>
      <td>9.99</td>
      <td>10,000,000 .. 20,000,000</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>236.0</td>
      <td>128.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>Counter-Strike: Condition Zero</td>
      <td>Mar 1, 2004</td>
      <td>9.99</td>
      <td>10,000,000 .. 20,000,000</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>Half-Life 2: Deathmatch</td>
      <td>Nov 1, 2004</td>
      <td>4.99</td>
      <td>5,000,000 .. 10,000,000</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>Half-Life: Source</td>
      <td>Jun 1, 2004</td>
      <td>9.99</td>
      <td>2,000,000 .. 5,000,000</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
additional = additional.dropna(subset = ['owners', 'release_date'])
additional = additional.reset_index(drop = True)
additional['Critic_Score'] = additional['metascore']/10
additional['Year'] = additional['release_date']
additional['owners'] = additional['owners'].astype(str)
for i in range(len(additional)):
    str(additional.loc[i, 'owners'])
    nums = additional.loc[i, 'owners'].split('\xa0..\xa0')
#     print(nums)
    additional.loc[i, 'owners'] = float((locale.atoi(nums[1]) - locale.atoi(nums[0])) / 2000000)
#     print(additional.loc[i, 'owners'])
    temp = additional.loc[i, 'Year'].split(', ')
    if len(temp) != 2:
        additional.loc[i, 'Year'] = np.nan
    else:
        additional.loc[i, 'Year'] = int(temp[1])
#     print(additional.loc[i, 'Year'].split(', ')[1])
additional = additional.dropna(subset = ['release_date'])
additional = additional.drop(columns = ['number', 'price', 'average_playtime', 'median_playtime'])
additional.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game</th>
      <th>release_date</th>
      <th>owners</th>
      <th>developer</th>
      <th>publisher</th>
      <th>metascore</th>
      <th>Critic_Score</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Half-Life 2</td>
      <td>Nov 16, 2004</td>
      <td>5</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>96.0</td>
      <td>9.6</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Counter-Strike: Source</td>
      <td>Nov 1, 2004</td>
      <td>5</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>88.0</td>
      <td>8.8</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Counter-Strike: Condition Zero</td>
      <td>Mar 1, 2004</td>
      <td>5</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>65.0</td>
      <td>6.5</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Half-Life 2: Deathmatch</td>
      <td>Nov 1, 2004</td>
      <td>2.5</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Half-Life: Source</td>
      <td>Jun 1, 2004</td>
      <td>1.5</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004</td>
    </tr>
  </tbody>
</table>
</div>




```python
additional['Name'] = additional['game']
additional['Developer'] = additional['developer']
additional['Global_Sales'] = additional['owners']
df = df.dropna(subset = ['Year'])
df['Year'] = df['Year'].astype(int)

additional = additional.drop(columns=['metascore', 'release_date', 'publisher', 'game', 'developer', 'owners'])
df = df.drop(columns=['Rank', 'basename', 'Total_Shipped', 'Platform', 'Publisher', 'VGChartz_Score', 
                      'Last_Update', 'url', 'status', 'Vgchartzscore', 'img_url',  'User_Score'])
pd.merge(df, additional, on = ['Name', 'Year'] , how = 'left')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Genre</th>
      <th>ESRB_Rating</th>
      <th>Developer</th>
      <th>Critic_Score</th>
      <th>Global_Sales</th>
      <th>NA_Sales</th>
      <th>PAL_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Sports</td>
      <td>E</td>
      <td>Nintendo EAD</td>
      <td>7.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>Platform</td>
      <td>NaN</td>
      <td>Nintendo EAD</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Racing</td>
      <td>E</td>
      <td>Nintendo EAD</td>
      <td>8.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PlayerUnknown's Battlegrounds</td>
      <td>Shooter</td>
      <td>NaN</td>
      <td>PUBG Corporation</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wii Sports Resort</td>
      <td>Sports</td>
      <td>E</td>
      <td>Nintendo EAD</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.dropna(subset=['Developer', 'Genre'])
df = df.reset_index(drop = True)
df['Sales_Ranking'] = df['Global_Sales']
for i in range(len(df)):
    df.loc[i, 'Developer'] = str(df.loc[i, 'Developer'])
    if df.loc[i, 'Sales_Ranking'] >= 10:
        df.loc[i, 'Sales_Ranking'] = 4
    elif df.loc[i, 'Sales_Ranking'] >= 5 and df.loc[i, 'Sales_Ranking'] < 10:
        df.loc[i, 'Sales_Ranking'] = 3
    elif df.loc[i, 'Sales_Ranking'] >= 1 and df.loc[i, 'Sales_Ranking'] < 5:
        df.loc[i, 'Sales_Ranking'] = 2
    else:
        df.loc[i, 'Sales_Ranking'] = 1
le = LabelEncoder()
# ohe = OneHotEncoder(handle_unknown = 'ignore')
df['Sales_Ranking'] = df['Sales_Ranking'].astype(int)
# df['Developer'] = le.fit_transform(df['Developer'])
df['Genre'] = le.fit_transform(df['Genre'])
df = df.dropna(subset=['Global_Sales', 'ESRB_Rating'])
df['ESRB_Rating'] = le.fit_transform(df['ESRB_Rating'])
# df_temp = pd.DataFrame(ohe.fit_transform(df[['Genre']]).toarray())
```


```python

# df = df.join(df_temp)

df = df.reset_index(drop = True)
df = df[df['Global_Sales'] != 0.0]
df_for_visualization = df
```


```python
df_for_training = df
df_for_training = df.drop(columns = ['Name', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score'])
df_for_training = df_for_training.dropna(subset = ['NA_Sales'])
temp_df = df_for_training.drop(columns=['Genre', 'ESRB_Rating', 'Developer', 'Year', 'Sales_Ranking'])

x = temp_df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
temp_df = pd.DataFrame(x_scaled)
```


```python
df_for_training['Global_Sales'] = temp_df[0]
df_for_training['NA_Sales'] = temp_df[1]
df_for_training = df_for_training.dropna(subset = ['Global_Sales', 'NA_Sales'])
df_for_training = df_for_training.reset_index(drop = True)
df_for_training['Developer'] = le.fit_transform(df_for_training['Developer'])
# Shuffle and reorder the dataframe
df_for_training = df_for_training.sample(frac=1)[['Genre','ESRB_Rating','Developer','NA_Sales','Year','Global_Sales','Sales_Ranking']]
df_for_training
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>ESRB_Rating</th>
      <th>Developer</th>
      <th>NA_Sales</th>
      <th>Year</th>
      <th>Global_Sales</th>
      <th>Sales_Ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9592</th>
      <td>12</td>
      <td>5</td>
      <td>1355</td>
      <td>0.005123</td>
      <td>2002</td>
      <td>0.002954</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>15</td>
      <td>5</td>
      <td>1570</td>
      <td>0.047131</td>
      <td>2006</td>
      <td>0.047267</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>0</td>
      <td>3</td>
      <td>788</td>
      <td>0.024590</td>
      <td>2017</td>
      <td>0.037912</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>13</td>
      <td>5</td>
      <td>926</td>
      <td>0.017418</td>
      <td>2005</td>
      <td>0.032004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8433</th>
      <td>16</td>
      <td>0</td>
      <td>1763</td>
      <td>0.010246</td>
      <td>2009</td>
      <td>0.004924</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2032</th>
      <td>17</td>
      <td>0</td>
      <td>57</td>
      <td>0.038934</td>
      <td>2002</td>
      <td>0.037420</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10905</th>
      <td>10</td>
      <td>0</td>
      <td>126</td>
      <td>0.003074</td>
      <td>2004</td>
      <td>0.001477</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1121</th>
      <td>17</td>
      <td>0</td>
      <td>492</td>
      <td>0.013320</td>
      <td>2012</td>
      <td>0.061054</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6280</th>
      <td>1</td>
      <td>1</td>
      <td>181</td>
      <td>0.011270</td>
      <td>2015</td>
      <td>0.009355</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7248</th>
      <td>10</td>
      <td>0</td>
      <td>89</td>
      <td>0.014344</td>
      <td>2009</td>
      <td>0.006893</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>11348 rows × 7 columns</p>
</div>




```python
### 4.b Data Analysis and Visualization' <a name="data-ana-vis"></a>
```

## 5. Machine Learning Model <a name="ml-model"></a>
In this section, we are going to implement several models and predict global sales. In the world of machine learning, people
can split datas into two groups: numerical data and categorical data. Numerical data is everything that represented by numbers (integer
and floating point). It's continuous. Categorical data, however, is discrete. Different models will be used to predict these two type of data.

It is obvious to predict sales as numerical data but we have the accuracy concern(we will see accuracy in the **Result Analysis and Demonstration** section)
since the data may not demonstrate a strong linear trend. Therefore, we hope to predict it as categorical data: sale score is divided into 4 categories.
Games in ">10" category are expected to sell so greate that its name will left in history -- Grand Theft Auto, Pokemon, Call of duty and etc. You name it.
Games in "5-10" category are sold less than the top ones, but they are still great games. "5-1" games are good games. there are still large amount of customer want to
put them into their gaming library. The rest of games can be put into "1-0" categories. We respect the efforts that game developers put into them but they are relatively
niche.
### 5.a What and Why <a name="what-why"></a>

We want to use *multiple linear regression* for predicting numerical sale number. The reason is that we intend to investigate
how strong the relationship is between many independent variables (in this case, critic score, developers and other variables) and
one dependent variable -- sale score. We made several assumptions for using multiple linear regression.
 - Homogeneity of Variance: the size of the error in our prediction doesn't change a lot
 - Independence of Observations: each game is independent of others.
 - Linearity: the line of best fit through the data point is a straight line.

Several models will be used for the prediction of categorical sale number: *Random forest*, *k-nearest neighbors* (KNN) and
*Support vector machine*(SVM)

Single decision tree suffers from a high variance, which makes them less accurate than other models. However, random forest fixes
this problem. Benefits of using random forests:
 -  Bagging and bootstrap reduce the output variance
 -  Able to handle large dataset with high dimensionality (which is our datset)

k-nearest neightbors, as one of the most famous classifications algorithm, surely have many positive sides:
 - No training period
 - Easily to add new data
 - Easy to implement

Here is the advantages of choosing support vector machine as one of our algorithem.
 -  Effective in high dimensional spaces
 -  Use a subset of training set in the decision function and, therefore, prevent overfitting
 -  Memory efficient

### 5.b Training <a name="training"></a>
**Multiple Linear Regression**
We will use sklearn library for most of our training task. Non-linear regression is little bit tricky and we wish to use scipy library for training.


```python
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier
import sklearn

# build model for numerical predictors
muti_linear_regression = linear_model.LinearRegression(n_jobs=-1)
```

Explanation:

This is a very simple and straight-forward model with n_jobs = -1, which means we want to use all available CPU cores for efficiency purpose


```python
# build model for categorical predictors
random_forest = RandomForestClassifier(n_estimators = 1000, random_state=42,max_depth=4,n_jobs = -1)
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
svm = sklearn.svm.LinearSVC(max_iter=2000,dual=False)
```

Explanation:

To determine the number of trees (n_estimators in the function), we theoretically want as many trees as possible but the
margin of accuracy of getting more than 1000 trees become neglectable. random_state will increase the randomness when the algorithm is
bootstrapping.It is suggested that the maximum depth of the tree is sqrt(number of features), and also the
more depth of a tree, the better it perform with diminishing returns. I will just choose 4 and the benefit of more than 4 is too small. The number of jobs indicates how many
threads that are working in parallel.

As for kNN, to determine the number of neighbors, I did several experiments. It turns out that n_neightbors = 5 can generate best output. Too small n_neightbor will result in
unstable decision boundaries will too large will make the decision boundaries unclear.

SVM is little bit intriguing. There are two options for us to set the "decision_function_shape". One is "ovo", which stands for one-verses-one, and the other option is called one-vs-the-rest.
One-verse-one compare each classcifier with the predict value one by one while the one verse the rest option treats the x as a group and compare it with the y. In our case, we consider all the regressor
as a group. The reason why we set max_iter to 2000 is that it will not converge at default number of iterations


```python
# Assign first several columns as X and last two columns as ground truth
X = df_for_training.iloc[:, 0:5]
y_categorical = df_for_training[['Sales_Ranking']].to_numpy().flatten()
y_numerical = df_for_training[['Global_Sales']].to_numpy().flatten()
```


```python
# numerical model
# 10-fold cross validation for multi-linear regression:
linear_score = []
X = X.to_numpy()
for train_index, test_index in model_selection.KFold(n_splits=10).split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_numerical[train_index], y_numerical[test_index]
    model = muti_linear_regression.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    linear_score.append(score)
print('The average score for linear regression is ',np.average(linear_score))
print("The standard error of the score is ", np.std(linear_score))
```

    The average score for linear regression is  0.8389972878343632
    The standard error of the score is  0.02656213559835733
    


```python
# categorical model
# Implement 10-fold cross validation
rfr_score = model_selection.cross_val_score(random_forest, X, y_categorical, cv = 10)
print("The average score for Random Forest is ", np.average(rfr_score))
print("The standard error of the score is ", np.std(rfr_score))
knn_score = model_selection.cross_val_score(knn, X, y_categorical, cv = 10)
print("The average score for kNN is ", np.average(knn_score))
print("The standard error of the score is ", np.std(knn_score))
svm_score = model_selection.cross_val_score(svm,X, y_categorical, cv = 10)
print("The average score for SVM is ", np.average(svm_score))
print("The standard error of the score is ", np.std(svm_score))
```

    The average score for Random Forest is  0.9455400943212984
    The standard error of the score is  0.005250425986782506
    The average score for kNN is  0.8735466828271526
    The standard error of the score is  0.007593598713356333
    The average score for SVM is  0.8983963048427072
    The standard error of the score is  0.01175409018656301
    

### Result Anlysis and Demonstration <a name="result-and-demon"></a>
Below is the bar graph of accuracy score for different models


```python
import matplotlib.pyplot as plt
models = ['linear_reg', 'radom forest', 'knn', 'svm']
scores = [linear_score,rfr_score, knn_score,svm_score]
accuracy = np.average(scores,axis=1)
std = np.std(scores,axis=1)
fig, ax = plt.subplots()
ax.bar(models,accuracy,align = 'center',yerr = std, capsize=20)
ax.set_xticks(models)
ax.set_title('Accuracy of Four Models For Game Sale Prediction')
ax.yaxis.grid(True)
ax.set_xlabel("Model")
ax.set_ylabel('Accuracy Score')
plt.show()
```


    
![png](README_files/README_24_0.png)
    


Random forest model has the best accuracy score and I think bagging and bootstrap could be the reason why it outperformed other models.
Also, the prediction for categorical variable generally better than the numerical prediction becuase, intuitively, predicting a category is
easier than a specific number.

Since we are interested in the difference between two variables for the same subject, we are going to perform paired-t test for the predicted value and the ground truth to see
the statistical difference between them. Our null hypothesis would be the average difference between the predicted value and ground truth is 0 and alternative hypothesis is the
average difference is not 0. We choose alpha value = 0.05


```python
from scipy import stats
muti_linear_regression.fit(X,y_categorical)
pred_y = muti_linear_regression.predict(X)
print("paired t-test for random multi-linear regression is \n")
stats.ttest_rel(y_categorical, pred_y)
```

    paired t-test for random multi-linear regression is 
    
    




    Ttest_relResult(statistic=1.2374634887003975e-13, pvalue=0.9999999999999013)




```python
random_forest.fit(X,y_categorical)
pred_y = random_forest.predict(X)
print("paired t-test for random forest result is \n")
stats.ttest_rel(y_categorical, pred_y)
```

    paired t-test for random forest result is 
    
    




    Ttest_relResult(statistic=9.064022064910278, pvalue=1.4630546520490245e-19)




```python
knn.fit(X,y_categorical)
pred_y = knn.predict(X)
print("paired t-test for k nearest neighbor result is \n")
stats.ttest_rel(y_categorical, pred_y)
```

    paired t-test for k nearest neighbor result is 
    
    




    Ttest_relResult(statistic=23.45994627419661, pvalue=6.88479830215365e-119)




```python
svm.fit(X,y_categorical)
pred_y = svm.predict(X)
print("paired t-test for support vector machine result is \n")
stats.ttest_rel(y_categorical, pred_y)
```

    paired t-test for support vector machine result is 
    
    




    Ttest_relResult(statistic=39.954277306310075, pvalue=0.0)



From above result, it is interesting to see that we failed reject null hypothesis (i.e. there is no difference between
the predicted value and ground truth for multi-linear regression paired-t test) but reject the null hypothesis (that is, there IS a difference)
for the rest of three paired-t test. However, according to the accuracy score, random forest model achieved the highest. Why does this happen?

According the formula that calculate t-value, we need to find the standard deviation of the difference between two groups. This standard deviation doesn't
make sense when it comes to category. You can think it as using l2 loss (mean squared error) instead of cross-entropy loss for categorical problem. Therefore,
we'd better directly use accuracy score for model-model comparison.

## 6. Future Application <a name="future-app"></a>
TODO:
## 7. Reference and External Link <a name="ref-and-extlink"></a>
#### Want to to know more about multiple linear regression?
 - https://www.scribbr.com/statistics/multiple-linear-regression/
 - https://en.wikipedia.org/wiki/Linear_regression
 - https://towardsdatascience.com/understanding-multiple-regression-249b16bde83e

#### Extend materials for support vector machine, Knn, random forest
 - https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
 - https://www.youtube.com/watch?v=1NxnPkZM9bc
 - https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
 - https://scikit-learn.org/stable/modules/svm.html
 - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#### paired-t test reading:
 - https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/SAS/SAS4-OneSampleTtest/SAS4-OneSampleTtest7.html
