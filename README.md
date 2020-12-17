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
In this section, we will load and process the two datasets, "vgsales-12-4-2019.csv" which is our main dataset and "video_games.csv" which is an additional dataset we will use to fill in some important missing values like "Global_Sales" and "Critic_Score" in the first dataset.
#### Datasets
"vgsales-12-4-2019.csv" is a dataset with 55,792 records of game sales collected by 2019, it is loaded from Kaggle. There are missing values in the column "Global_Sales", since this column is important in our analysis, we load another dataset "video_games.csv" to fill in these values as many as possbile.

"video_games.csv" is a dataset of Steam game sales, we load it from [Github](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-07-30). It has a cloumn "owners" which includes a range of the number of players that own the game, we take the median of this range as the value to replace NaN in the column "Global_Sales", and we will do the same for the missing values in "Critic_Score". To merge the two datasets after doing necessary processing, we will use a LEFT JOIN on the columns "Name" and "Year".
### 4.a. Load and Clean Data <a name="load-and-clean"></a>
In the following cell we import the libraries we will use for data preprocessing, then we load the datasets using pandas.read_csv() function and create a preview of the datasets using df.head() function.


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
      <th>Rank</th>
      <th>Name</th>
      <th>basename</th>
      <th>Genre</th>
      <th>ESRB_Rating</th>
      <th>Platform</th>
      <th>Publisher</th>
      <th>Developer</th>
      <th>VGChartz_Score</th>
      <th>Critic_Score</th>
      <th>...</th>
      <th>NA_Sales</th>
      <th>PAL_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Year</th>
      <th>Last_Update</th>
      <th>url</th>
      <th>status</th>
      <th>Vgchartzscore</th>
      <th>img_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Wii Sports</td>
      <td>wii-sports</td>
      <td>Sports</td>
      <td>E</td>
      <td>Wii</td>
      <td>Nintendo</td>
      <td>Nintendo EAD</td>
      <td>NaN</td>
      <td>7.7</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2006.0</td>
      <td>NaN</td>
      <td>http://www.vgchartz.com/game/2667/wii-sports/?...</td>
      <td>1</td>
      <td>NaN</td>
      <td>/games/boxart/full_2258645AmericaFrontccc.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Super Mario Bros.</td>
      <td>super-mario-bros</td>
      <td>Platform</td>
      <td>NaN</td>
      <td>NES</td>
      <td>Nintendo</td>
      <td>Nintendo EAD</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1985.0</td>
      <td>NaN</td>
      <td>http://www.vgchartz.com/game/6455/super-mario-...</td>
      <td>1</td>
      <td>NaN</td>
      <td>/games/boxart/8972270ccc.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Mario Kart Wii</td>
      <td>mario-kart-wii</td>
      <td>Racing</td>
      <td>E</td>
      <td>Wii</td>
      <td>Nintendo</td>
      <td>Nintendo EAD</td>
      <td>NaN</td>
      <td>8.2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2008.0</td>
      <td>11th Apr 18</td>
      <td>http://www.vgchartz.com/game/6968/mario-kart-w...</td>
      <td>1</td>
      <td>8.7</td>
      <td>/games/boxart/full_8932480AmericaFrontccc.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>PlayerUnknown's Battlegrounds</td>
      <td>playerunknowns-battlegrounds</td>
      <td>Shooter</td>
      <td>NaN</td>
      <td>PC</td>
      <td>PUBG Corporation</td>
      <td>PUBG Corporation</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017.0</td>
      <td>13th Nov 18</td>
      <td>http://www.vgchartz.com/game/215988/playerunkn...</td>
      <td>1</td>
      <td>NaN</td>
      <td>/games/boxart/full_8052843AmericaFrontccc.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Wii Sports Resort</td>
      <td>wii-sports-resort</td>
      <td>Sports</td>
      <td>E</td>
      <td>Wii</td>
      <td>Nintendo</td>
      <td>Nintendo EAD</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009.0</td>
      <td>NaN</td>
      <td>http://www.vgchartz.com/game/24656/wii-sports-...</td>
      <td>1</td>
      <td>8.8</td>
      <td>/games/boxart/full_7295041AmericaFrontccc.jpg</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
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



#### Clean the additional dataset
Now we process the additional dataset. First, we drop rows with NaN values in the columns "owners" and "release_date", and reset the index of the dataframe. Next, we divide the values in the column "metascore" by 10 to make the unit match in both tables, and store the results in a new column "Critic_Score". To calculate the median of the range of owners, we convert the values in column "owners" to string, then iterate through the dataframe to split the string and convert the results to integers, finally we calculate the result (in millions). In the same loop, we also extract the value of year from the column "release_date". Note that there are NaN values in the column "Critic_Score" but we do not drop them because our main goal is to get more values for "Global_Sales". After renaming the columns that we will use to merge the datasets (by copying to new columns and dropping the original columns), we finish processing the additional dataset.


```python
additional = additional.dropna(subset = ['owners', 'release_date'])
additional = additional.reset_index(drop = True)
additional['Critic_Score'] = additional['metascore']/10
additional['Year'] = additional['release_date']
additional['owners'] = additional['owners'].astype(str)
# calculate median of owners and extract value of year
for i in range(len(additional)):
    str(additional.loc[i, 'owners'])
    nums = additional.loc[i, 'owners'].split('\xa0..\xa0')
    additional.loc[i, 'owners'] = float((locale.atoi(nums[1]) - locale.atoi(nums[0])) / 2000000)
    temp = additional.loc[i, 'Year'].split(', ')
    if len(temp) != 2:
        additional.loc[i, 'Year'] = np.nan
    else:
        additional.loc[i, 'Year'] = int(temp[1])

additional = additional.dropna(subset = ['release_date'])
# drop useless columns and rename columns
additional = additional.drop(columns = ['number', 'price', 'average_playtime', 'median_playtime'])
additional['Name'] = additional['game']
additional['Developer'] = additional['developer']
additional['Global_Sales'] = additional['owners']
additional = additional.drop(columns=['metascore', 'release_date', 'publisher', 'game', 'developer', 'owners'])
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
      <th>Critic_Score</th>
      <th>Year</th>
      <th>Name</th>
      <th>Developer</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.6</td>
      <td>2004</td>
      <td>Half-Life 2</td>
      <td>Valve</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.8</td>
      <td>2004</td>
      <td>Counter-Strike: Source</td>
      <td>Valve</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.5</td>
      <td>2004</td>
      <td>Counter-Strike: Condition Zero</td>
      <td>Valve</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>2004</td>
      <td>Half-Life 2: Deathmatch</td>
      <td>Valve</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2004</td>
      <td>Half-Life: Source</td>
      <td>Valve</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



#### Merge the main dataset and the additional dataset
Before merging the two datasets, we drop rows having missing values in the column "Year" as we will use this column and the column "Name" in merging. Then we drop columns we will not use in data visualization and data analysis, which are 'Rank', 'basename', 'Total_Shipped', 'Platform', 'Publisher', 'VGChartz_Score', 'Last_Update', 'url', 'status', 'Vgchartzscore', 'img_url', 'User_Score'. 

The type of join we choose is left join, as we do not want to add excessive records from the additional dataset.


```python
df = df.dropna(subset = ['Year'])
df['Year'] = df['Year'].astype(int)
df = df.drop(columns=['Rank', 'basename', 'Total_Shipped', 'Platform', 'Publisher', 'VGChartz_Score', 
                      'Last_Update', 'url', 'status', 'Vgchartzscore', 'img_url',  'User_Score'])
# left join on 'Name', 'Year'
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



#### Process the merged dataset
First we drop rows with missing values in the columns "Developer" and "Genre", and reset the index. We create a new column "Sales_Ranking" referring to a new category, when a game has over 10 million sales, its Sales_Ranking is 4, a game with 5-10 million sales has a Sales_Ranking of 3, a game with 1-5 million sales has a Sales_Ranking of 2, a game with sales lower than 1 million will have Sales_Ranking of 1.

For data analysis, we need to convert categorical variable to numerical variable. We choose to use label encoding on the three categories, "Genre", "ESRB_Rating" and "Developer". Since the total number of developers is large and we will use "Genre" in data visualization, we will process "Developer" and "Genre" later. To create the dataframe for data analysis, we need to drop columns that we will not use, which are 'Name', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score'. Also, we will normalize the numerical values in the columns "Global_Sales" and "NA_Sales" by using the sklearn.preprocessing module.


```python
df = df.dropna(subset=['Developer', 'Genre'])
df = df.reset_index(drop = True)
df['Sales_Ranking'] = df['Global_Sales']
# get values for the column 'Sales_Ranking'
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
df = df.dropna(subset=['Global_Sales', 'ESRB_Rating'])
df['ESRB_Rating'] = le.fit_transform(df['ESRB_Rating'])
# df_temp = pd.DataFrame(ohe.fit_transform(df[['Genre']]).toarray())
# df = df.join(df_temp)
```

#### Dataframe for data visualization
Now we create the dataframe for data visualization


```python
df = df.reset_index(drop = True)
df = df[df['Global_Sales'] != 0.0]
df_for_visualization = df.dropna(subset = ['Global_Sales', 'NA_Sales'])
df_for_visualization = df_for_visualization.reset_index(drop = True)
df_for_visualization.head()
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
      <th>Sales_Ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>Action</td>
      <td>3</td>
      <td>Rockstar North</td>
      <td>9.4</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>9.85</td>
      <td>0.99</td>
      <td>3.12</td>
      <td>2013</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grand Theft Auto V</td>
      <td>Action</td>
      <td>3</td>
      <td>Rockstar North</td>
      <td>9.7</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>9.71</td>
      <td>0.60</td>
      <td>3.02</td>
      <td>2014</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grand Theft Auto: Vice City</td>
      <td>Action</td>
      <td>3</td>
      <td>Rockstar North</td>
      <td>9.6</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>5.49</td>
      <td>0.47</td>
      <td>1.78</td>
      <td>2002</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grand Theft Auto V</td>
      <td>Action</td>
      <td>3</td>
      <td>Rockstar North</td>
      <td>NaN</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>5.33</td>
      <td>0.06</td>
      <td>1.42</td>
      <td>2013</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>Shooter</td>
      <td>3</td>
      <td>Treyarch</td>
      <td>NaN</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>6.05</td>
      <td>0.41</td>
      <td>2.44</td>
      <td>2015</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### Dataframe for data analysis
To create the dataframe for data analysis, we need to drop columns that we will not use, which are 'Name', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score'. Also, we will normalize the numerical values in the columns "Global_Sales" and "NA_Sales" by using the sklearn.preprocessing module.


```python
df_for_training = df.drop(columns = ['Name', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score'])
df_for_training = df_for_training.dropna(subset = ['NA_Sales'])
temp_df = df_for_training.drop(columns=['Genre', 'ESRB_Rating', 'Developer', 'Year', 'Sales_Ranking'])
# create a temp dataframe for normalization of numerical values
x = temp_df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
temp_df = pd.DataFrame(x_scaled)
# copy normalized values into the training dataframe
df_for_training['Global_Sales'] = temp_df[0]
df_for_training['NA_Sales'] = temp_df[1]
df_for_training = df_for_training.dropna(subset = ['Global_Sales', 'NA_Sales'])
df_for_training = df_for_training.reset_index(drop = True)
# apply label encoding on 'Genre' and 'Developer'
df_for_training['Genre'] = le.fit_transform(df_for_training['Genre'])
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
      <th>4617</th>
      <td>12</td>
      <td>0</td>
      <td>1546</td>
      <td>0.023566</td>
      <td>1997</td>
      <td>0.014771</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9879</th>
      <td>11</td>
      <td>0</td>
      <td>1313</td>
      <td>0.005123</td>
      <td>1996</td>
      <td>0.001969</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3457</th>
      <td>17</td>
      <td>0</td>
      <td>905</td>
      <td>0.043033</td>
      <td>1996</td>
      <td>0.021664</td>
      <td>1</td>
    </tr>
    <tr>
      <th>475</th>
      <td>2</td>
      <td>5</td>
      <td>1772</td>
      <td>0.090164</td>
      <td>2003</td>
      <td>0.108813</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>0</td>
      <td>0</td>
      <td>1813</td>
      <td>0.023566</td>
      <td>2004</td>
      <td>0.023634</td>
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
      <th>1642</th>
      <td>17</td>
      <td>0</td>
      <td>505</td>
      <td>0.044057</td>
      <td>2013</td>
      <td>0.044806</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6082</th>
      <td>18</td>
      <td>1</td>
      <td>703</td>
      <td>0.019467</td>
      <td>2008</td>
      <td>0.009355</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7928</th>
      <td>10</td>
      <td>1</td>
      <td>127</td>
      <td>0.009221</td>
      <td>2006</td>
      <td>0.004924</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6757</th>
      <td>16</td>
      <td>1</td>
      <td>223</td>
      <td>0.011270</td>
      <td>2008</td>
      <td>0.007386</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10274</th>
      <td>10</td>
      <td>5</td>
      <td>477</td>
      <td>0.004098</td>
      <td>2002</td>
      <td>0.001477</td>
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


    
![png](README_files/README_29_0.png)
    


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


```python
knn.fit(X,y_categorical)
pred_y = knn.predict(X)
print("paired t-test for k nearest neighbor result is \n")
stats.ttest_rel(y_categorical, pred_y)
```


```python
svm.fit(X,y_categorical)
pred_y = svm.predict(X)
print("paired t-test for support vector machine result is \n")
stats.ttest_rel(y_categorical, pred_y)
```

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


```python
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
