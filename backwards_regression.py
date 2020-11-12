import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import statsmodels.api as sm
from sklearn import preprocessing
#---------------------------------
# Feature Engineering Functions 
#---------------------------------
def regionn(n): 
	regions = {'Albany':1, 'Atlanta' : 2, 'BaltimoreWashington':3, 'Boise':4, 'Boston':5,'BuffaloRochester':6, 'California':7, 'Charlotte':8, 'Chicago':9, 'CincinnatiDayton':10,'Columbus':11,'DallasFtWorth':12, 'Denver':13, 'Detroit':14, 'GrandRapids':15, 'GreatLakes':16,'HarrisburgScranton':17, 'HartfordSpringfield':18, 'Houston':19, 'Indianapolis':20,'Jacksonville':21, 'LasVegas':22, 'LosAngeles':23, 'Louisville':24, 'MiamiFtLauderdale':25,'Midsouth':26, 'Nashville':27, 'NewOrleansMobile':28, 'NewYork':29 ,'Northeast':30,'NorthernNewEngland':31, 'Orlando':32, 'Philadelphia':33, 'PhoenixTucson':34,'Pittsburgh':35, 'Plains':36, 'Portland':37, 'RaleighGreensboro':38, 'RichmondNorfolk':39,'Roanoke':40, 'Sacramento':41, 'SanDiego':42, 'SanFrancisco':43, 'Seattle':44,'SouthCarolina':45, 'SouthCentral':46 ,'Southeast':47, 'Spokane':48, 'StLouis':49, 'Syracuse':50, 'Tampa':51, 'TotalUS':52, 'West':53, 'WestTexNewMexico':54 }
	return regions[n] 
def total_revenue(x, y):
	return x * y

#--------------------------------
#DATA PROCESSING & PRE-PROCESSING
#--------------------------------
df = pd.read_csv('avocado.csv')
 
df.rename(columns={'Total Volume':'TV'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

df['month'] = pd.DatetimeIndex(df['Date']).month 

df['region_number']  = df.apply(lambda row: regionn(row.region), axis = 1)

#split data set into both conventional and organic 
organic = df[df["type"] == "organic"]
test2 = df[df["type"] == "conventional"]
test2 = test2[test2["region"] == "SanDiego"]
test2 = test2.sort_values(by=['Date']) 
sandiego_X = test2[['month','year','XLarge Bags', 'Large Bags', 'Small Bags', 'Total Bags', 'TV', '4046', '4225', '4770']]
# sandiego_X = test2[['month','year','XLarge Bags', 'Large Bags', 'Total Bags', 'TV', '4046', '4770']]
print(test2)
#
from sklearn import preprocessing# Get column names first
names = sandiego_X.columns# Create the Scaler object
scaler = preprocessing.StandardScaler()# Fit your data on the scaler object
scaled_df = scaler.fit_transform(sandiego_X)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df)

