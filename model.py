import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_and_save():
    # Load dataset
    df = pd.read_csv('Marketing_Data_Clean.csv')

    # Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day'] = df['Date'].dt.day

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Platform', 'Campaign_Name'])

    # Define Features and Target
    # We remove 'Revenue_INR' (Target) and 'ROI_%' (Leakage)
    X = df_encoded.drop(['Date', 'Revenue_INR', 'ROI_%'], axis=1)
    y = df_encoded['Revenue_INR']

    # Save column names to ensure app.py matches this exact structure
    model_columns = list(X.columns)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save files
    joblib.dump(model, 'marketing_model.pkl')
    joblib.dump(model_columns, 'model_features.pkl')
    print("✅ Model and Features saved successfully!")


if __name__ == "__main__":
    train_and_save()