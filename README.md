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
    c. [Result Anlysis and Demonstration](#result-ana-demon)
6. [Future Application](#future-app)
7. [Reference and External Link](#ref-and-extlink)

## 1. Introduction <a name="introduction"></a>
TODO:

## 2. Install Packages <a name="install-pkg"></a>
```
pip install kaggle numpy matplotlib pandas
```
or use environment.yml to install packages in conda environment
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
### 4.a Load and Clean Data <a name="load-and-clean"></a>


```python
#TODO:
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("vgsales-12-4-2019.csv")
df = df.drop(columns=['basename', 'Total_Shipped', 'Platform', 'Publisher', 'VGChartz_Score', 'Last_Update', 'url', 'status', 'Vgchartzscore', 'img_url', 'ESRB_Rating'])
df = df.dropna(subset=['NA_Sales', 'Developer', 'Genre'])
df = df.reset_index()
for i in range(len(df)):
    df.loc[i, 'Developer'] = str(df.loc[i, 'Developer'])

le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown = 'ignore')
df['Developer'] = le.fit_transform(df['Developer'])
df['Genre'] = le.fit_transform(df['Genre'])
# df_genre = pd.DataFrame(ohe.fit_transform(df[['Genre']]).toarray())
# df_developer = pd.DataFrame(ohe.fit_transform(df[['Developer']]).toarray())
df_temp = pd.DataFrame(ohe.fit_transform(df[['Genre', 'Developer']]).toarray())
```


```python
df = df.join(df_temp)
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
      <th>index</th>
      <th>Rank</th>
      <th>Name</th>
      <th>Genre</th>
      <th>Developer</th>
      <th>Critic_Score</th>
      <th>User_Score</th>
      <th>Global_Sales</th>
      <th>NA_Sales</th>
      <th>PAL_Sales</th>
      <th>...</th>
      <th>2211</th>
      <th>2212</th>
      <th>2213</th>
      <th>2214</th>
      <th>2215</th>
      <th>2216</th>
      <th>2217</th>
      <th>2218</th>
      <th>2219</th>
      <th>2220</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>20</td>
      <td>Grand Theft Auto V</td>
      <td>0</td>
      <td>1577</td>
      <td>9.4</td>
      <td>NaN</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>9.85</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>21</td>
      <td>Grand Theft Auto V</td>
      <td>0</td>
      <td>1577</td>
      <td>9.7</td>
      <td>NaN</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>9.71</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>31</td>
      <td>Grand Theft Auto: Vice City</td>
      <td>0</td>
      <td>1577</td>
      <td>9.6</td>
      <td>NaN</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>5.49</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>33</td>
      <td>Grand Theft Auto V</td>
      <td>0</td>
      <td>1577</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>5.33</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>35</td>
      <td>Call of Duty: Black Ops 3</td>
      <td>15</td>
      <td>1954</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>6.05</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2234 columns</p>
</div>



### 4.b Data Analysis and Visualization' <a name="data-ana-vis"></a>


```python
#TODO:
```

## 5. Machine Learning Model <a name="ml-model"></a>
### 5.a What and Why <a name="what-why"></a>


```python
#TODO:
```

### 5.b Training <a name="training"></a>


```python
#TODO:
```

### 5.c Result Anlysis and Demonstration <a name="result-ana-demon"></a>
TODO:
## 6. Future Application <a name="future-app"></a>
TODO:
## 7. Reference and External Link <a name="ref-and-extlink"></a>
TODO:
