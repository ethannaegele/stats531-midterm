import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("data/DAX_2010-2020.csv")

# Clean date
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data = data.sort_values('Date')

# Clean Open column
data['Open'] = (
    data['Open']
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

data.set_index('Date', inplace=True)
data = data.asfreq('B')  # Business day frequency
data['Open'] = data['Open'].ffill()

# ARMA(2,0,2) model training
model = ARIMA(data['Open'], order=(4, 0, 4))
model_fit = model.fit()


print("AIC:", model_fit.aic)



#           MA(1)       MA(2)       MA(3)       MA(4)
# AR (1)    -3965.5     -3965.7     -3972.9     -3973.5
# AR (2)    -3962       -3966.6     -3972.8     -3971.7
# AR (3)    -3961.6     -3970.7     -3974.7     -3972.2
# AR (4)    -3972.8     -3974.8     -3972.4     -3968

