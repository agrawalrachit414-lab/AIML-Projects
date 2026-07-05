import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].hist(df['count'], bins=30, color='blue')
axes[0,0].set_title("Count Distribution")

hour_avg = df.groupby('hour')['count'].mean()
axes[0,1].plot(hour_avg.index, hour_avg.values, color='red')
axes[0,1].set_title("Avg Count vs Hour")

season_avg = df.groupby('season')['count'].mean()
axes[0,2].bar(season_avg.index, season_avg.values, color='green')
axes[0,2].set_title("Avg Count vs Season")

weather_avg = df.groupby('weather')['count'].mean()
axes[1,0].bar(weather_avg.index, weather_avg.values, color='purple')
axes[1,0].set_title("Avg Count vs Weather")

weekday_avg = df.groupby('weekday')['count'].mean()
axes[1,1].plot(weekday_avg.index, weekday_avg.values, color='orange')
axes[1,1].set_title("Avg Count vs Weekday")

month_avg = df.groupby('month')['count'].mean()
axes[1,2].plot(month_avg.index, month_avg.values, color='brown')
axes[1,2].set_title("Avg Count vs Month")

plt.tight_layout()
plt.show()

df_fe = df.copy()

df_fe['workingday'] = df_fe['workingday'].astype(str).map({'True':1, 'False':0})
df_fe['holiday'] = df_fe['holiday'].astype(str).map({'True':1, 'False':0})

df_fe = pd.get_dummies(df_fe, columns=['season', 'weather'], drop_first=True)

df_fe['hour_sin'] = np.sin(2*np.pi*df_fe['hour']/24)
df_fe['hour_cos'] = np.cos(2*np.pi*df_fe['hour']/24)

df_fe['month_sin'] = np.sin(2*np.pi*df_fe['month']/12)
df_fe['month_cos'] = np.cos(2*np.pi*df_fe['month']/12)

df_fe['temp_humidity'] = df_fe['temp'] * df_fe['humidity']
df_fe['working_hour'] = df_fe['workingday'] * df_fe['hour']

df_fe['lag_1'] = df_fe['count'].shift(1)
df_fe['lag_24'] = df_fe['count'].shift(24)

df_fe = df_fe.dropna()

df_fe['count_log'] = np.log1p(df_fe['count'])

X = df_fe.drop(['count', 'count_log'], axis=1)
y = df_fe['count_log']

split = int(len(df_fe)*0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

models = {
    "Linear": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=100),
    "GBR": GradientBoostingRegressor(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    preds_exp = np.expm1(preds)
    y_test_exp = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_exp, preds_exp))
    mae = mean_absolute_error(y_test_exp, preds_exp)
    r2 = r2_score(y_test_exp, preds_exp)

    print(name)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    print()

plt.figure(figsize=(8,6))
plt.plot(y_test_exp.values[:200], label="actual")
plt.plot(preds_exp[:200], label="pred")
plt.legend()
plt.title("Pred vs Actual")
plt.show()
