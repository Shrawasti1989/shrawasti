"""
Created on Thu Sep 12 22:43:17 2024

@author: Shrawasti Sahare
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Creating Lags
def preprocessing(Data):
    
    # Create lag features for water demand
    for i in range(1, 8):  # Lag by 1 to 2 days
        Data[f'Demand_Lag_{i}'] = Data['Demand'].shift(i)
        
    # Create lag features for temperature 
    for i in range(1, 2):  # Lag by 1 to 2 days
        Data[f'tavg_Lag_{i}'] = Data['Demand'].shift(i)
    
    Data.dropna(inplace=True)  # Drop rows with NaN after creating lag features
    
    Data['tavg'] = Data['tavg'].interpolate(method='linear')
    Data['wspd'] = Data['wspd'].interpolate(method='linear')
    
    return Data

# Cross-validation, training and diagnostic graphs
def CVtimeseries(data, k=5):
    data = preprocessing(data)  # Preprocess the data to create lag features
    
    features = [f'Demand_Lag_{i}' for i in range(1, 7)] #+ [f'tavg_Lag_{i}' for i in range(1, 2)]
    X = data[features]
    y = data['Demand']
    
    tscv = TimeSeriesSplit(n_splits=k)
    model = LinearRegression()
    
    mse_scores = []
    r2_scores = []
    
    all_predictions = []
    all_actuals = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
    
    print(f"Average MSE over {k} folds: {np.mean(mse_scores)}")
    print(f"Average R-squared over {k} folds: {np.mean(r2_scores)}")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_actuals, all_predictions, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.show()
    
    # Plot residuals
    residuals = np.array(all_actuals) - np.array(all_predictions)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()
    
    # Plot distribution of errors
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.show()

    # Plot R-squared values for each fold
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k+1), r2_scores, marker='o', linestyle='--')
    plt.xlabel('Fold')
    plt.ylabel('R-squared')
    plt.title('R-squared for Each Fold')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    return model

#  Predicting water demand for the next 7 days
def prediction(model, data):
    # Get the last row of data
    last_row = data.iloc[-1]
    future_predictions = []

    # Predicting for the next 7 days
    for i in range(7):
        # Prepare the features used in the model (Demand_Lag_1 to Demand_Lag_6)
        X_new = pd.DataFrame([last_row[[f'Demand_Lag_{i}' for i in range(1, 7)]]])
        
        # Predict the demand for the next day
        y_pred = model.predict(X_new)
        future_predictions.append(y_pred[0])
        
        # Shift lagged demand values for the next prediction
        for j in range(6, 1, -1):
            last_row[f'Demand_Lag_{j}'] = last_row[f'Demand_Lag_{j-1}']
        last_row['Demand_Lag_1'] = y_pred[0]  # Update Demand_Lag_1 with the predicted value

    # Create a DataFrame for future dates and predicted demand
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Demand': future_predictions})
    
    print("Next 7 Days Demand Prediction:")
    print(future_df)

    # Plot the future predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_df['Date'], future_df['Predicted_Demand'], marker='o', linestyle='--', color='b')
    plt.xlabel('Date')
    plt.ylabel('Predicted Demand')
    plt.title('Predicted Water Demand for Next 7 Days')
    plt.grid(True)
    plt.show()

    return future_df

# Main function
def main():
    df_demand = pd.read_excel('DemandData.xlsx')
    df_weather = pd.read_csv('birmingham_weather_july_2022.csv')

    df_demand['Date'] = pd.to_datetime(df_demand['Date'], format='%d/%m/%Y')
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d/%m/%Y')

    # Merge demand and weather data
    data = pd.merge(df_demand, df_weather, on='Date')
    
    # Perform Time Series cross-validation
    model = CVtimeseries(data, k=5)
    
    # Predict water demand for the next 7 days
    prediction(model, data)

# Run the main function
if __name__ == "__main__":
    main()
