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
### Load and Clean Data <a name="load-and-clean"></a>


```python
#TODO:
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
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
additional = additional.dropna(subset = ['owners'])
additional = additional.reset_index(drop = True)
additional['Critic_Score'] = additional['metascore']/10
for i in range(len(additional)):
    str(additional.loc[i, 'owners'])
    nums = additional.loc[i, 'owners'].split('\xa0..\xa0')
#     print(nums)
    additional.loc[i, 'owners'] = float((locale.atoi(nums[1]) - locale.atoi(nums[0])) / 2000000)
#     print(additional.loc[i, 'owners'])
```


```python
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
    </tr>
  </tbody>
</table>
</div>




```python
additional['Name'] = additional['game']
additional['Developer'] = additional['developer']
additional['Global_Sales'] = additional['owners']
additional = additional.drop(columns=['metascore', 'release_date', 'publisher', 'game', 'developer', 'owners'])
df = df.drop(columns=['Rank', 'basename', 'Total_Shipped', 'Platform', 'Publisher', 'VGChartz_Score', 
                      'Last_Update', 'url', 'status', 'Vgchartzscore', 'img_url', 'ESRB_Rating', 'Year'])
pd.merge(df, additional, on = ['Name', 'Developer'] , how = 'outer')
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
      <th>Developer</th>
      <th>Critic_Score</th>
      <th>User_Score</th>
      <th>Global_Sales</th>
      <th>NA_Sales</th>
      <th>PAL_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Sports</td>
      <td>Nintendo EAD</td>
      <td>7.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>Platform</td>
      <td>Nintendo EAD</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Racing</td>
      <td>Nintendo EAD</td>
      <td>8.2</td>
      <td>9.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PlayerUnknown's Battlegrounds</td>
      <td>Shooter</td>
      <td>PUBG Corporation</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wii Sports Resort</td>
      <td>Sports</td>
      <td>Nintendo EAD</td>
      <td>8.0</td>
      <td>8.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.dropna(subset=['Developer', 'Genre'])
df = df.reset_index(drop = True)
for i in range(len(df)):
    df.loc[i, 'Developer'] = str(df.loc[i, 'Developer'])
le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown = 'ignore')
df['Developer'] = le.fit_transform(df['Developer'])
df['Genre'] = le.fit_transform(df['Genre'])
df_temp = pd.DataFrame(ohe.fit_transform(df[['Genre']]).toarray())
```


```python
df = df.join(df_temp)
df = df.drop(columns = ['Genre', 'User_Score'])
df = df.dropna(subset=['Global_Sales'])
df = df.reset_index(drop = True)
df = df[df['Global_Sales'] != 0.0]
```


```python
df
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
      <th>Developer</th>
      <th>Critic_Score</th>
      <th>Global_Sales</th>
      <th>NA_Sales</th>
      <th>PAL_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>0</th>
      <th>1</th>
      <th>...</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>5783</td>
      <td>9.4</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>9.85</td>
      <td>0.99</td>
      <td>3.12</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>Grand Theft Auto V</td>
      <td>5783</td>
      <td>9.7</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>9.71</td>
      <td>0.60</td>
      <td>3.02</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>Grand Theft Auto: Vice City</td>
      <td>5783</td>
      <td>9.6</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>5.49</td>
      <td>0.47</td>
      <td>1.78</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>Grand Theft Auto V</td>
      <td>5783</td>
      <td>NaN</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>5.33</td>
      <td>0.06</td>
      <td>1.42</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>Call of Duty: Black Ops 3</td>
      <td>7108</td>
      <td>NaN</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>6.05</td>
      <td>0.41</td>
      <td>2.44</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18028</th>
      <td>FirePower for Microsoft Combat Flight Simulator 3</td>
      <td>6118</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18029</th>
      <td>Tom Clancy's Splinter Cell</td>
      <td>7205</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18030</th>
      <td>Ashita no Joe 2: The Anime Super Remix</td>
      <td>1225</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>18031</th>
      <td>Tokyo Yamanote Boys for V: Main Disc</td>
      <td>5684</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>18032</th>
      <td>NadePro!! Kisama no Seiyuu Yatte Miro!</td>
      <td>2932</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
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
<p>18033 rows × 28 columns</p>
</div>




```python
df.shape
```




    (12962, 30)



### Data Analysis and Visualization' <a name="data-ana-vis"></a>


```python
#TODO:
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
### What and Why <a name="what-why"></a>



### Training <a name="training"></a>


```python
#TODO:
```

### Result Anlysis and Demonstration <a name="result-ana-demon"></a>
TODO:
## 6. Future Application <a name="future-app"></a>
TODO:
## 7. Reference and External Link <a name="ref-and-extlink"></a>
TODO:
