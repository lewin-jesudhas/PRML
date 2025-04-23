# import pandas as pd

# # Path to your .data file
# data_path = r'abalone/abalone.data'

# # Define the column names (from UCI documentation)
# columns = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight',
#            'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']

# # Read the data
# df = pd.read_csv(data_path, header=None, names=columns)

# # Save as .csv
# df.to_csv("abalone.csv", index=False)

# print("Conversion complete: abalone.csv")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('abalone.csv')
df.head()

X=df[['Length']]
y=df['Rings']+1.5

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)


print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
tolerance = 1.5
accuracy = np.mean(np.abs(y_pred - y_test) <= tolerance) * 100
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print(f"Custom Accuracy (±1.5 years): {accuracy:.2f}%")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor

# # Load the data
# df = pd.read_csv('abalone.csv')

# # Features and target
# X = df[['Length']]
# y = df['Rings'] + 1.5  # Estimated age

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # (Optional) Feature scaling for XGBoost – not required, but you can still do it
# # If Length was the only feature, might help slightly
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize and train XGBoost model
# model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation metrics
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# tolerance = 1.5
# accuracy = np.mean(np.abs(y_pred - y_test) <= tolerance) * 100

# # Print results
# print("Mean Squared Error:", mse)
# print("R-squared:", r2)
# print(f"Custom Accuracy (±1.5 years): {accuracy:.2f}%")

# # Plot Actual vs Predicted
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted (XGBoost)")
# plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
# plt.show()
