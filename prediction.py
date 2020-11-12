import numpy as np 
import pandas as pd 
import warnings 
import regression as rg
warnings.filterwarnings("ignore")

avocados = pd.read_csv("avocado.csv")
# make "Date" readable and allow months to be readable
avocados['Date'] = pd.to_datetime(avocados['Date'])
avocados['month'] = pd.DatetimeIndex(avocados['Date']).month
avocados.rename(columns={'Total Volume':'TV'}, inplace=True)
# define organic and conventional
organic = avocados[avocados["type"] == "organic"]
conventional = avocados[avocados["type"] == "conventional"]
# define regions_type
sandiego_conv = conventional[conventional["region"] == "SanDiego"]

# ?
sandiego_conv = sandiego_conv.groupby([avocados['Date'].dt.date]).mean()
# normalize
# sandiego_conv = (sandiego_conv - sandiego_conv.mean())/sandiego_conv.std()

X = sandiego_conv[[ 'month','year','XLarge Bags', 'Large Bags',  'Total Bags', 'TV']].values
y = sandiego_conv['AveragePrice'].values

model = rg.OrdinaryLeastSquares()
model.fit(X, y)

y_preds = []
for row in X: 
    y_preds.append(model.predict(row))


print(pd.DataFrame({'Actual': y, 'Predicted': np.ravel(y_preds)}))
