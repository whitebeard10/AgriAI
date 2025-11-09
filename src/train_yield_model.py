import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib
import numpy as np

def train_yield_model(data_path='data/crop_yield.csv', model_path='models/yield_predictor.pkl'):
    """
    Trains a RandomForestRegressor model to predict crop yield and saves it.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop rows with NaN values for simplicity, or you could impute them
    df.dropna(inplace=True)

    # Define features (X) and target (y)
    # We will predict 'Yield'
    X = df.drop(['Yield', 'Production'], axis=1) # Drop Production as Yield is derived from it
    y = df['Yield']

    # Identify categorical and numerical features
    categorical_features = ['Crop', 'Season', 'State']
    numerical_features = X.columns.drop(categorical_features)

    # Create a preprocessor for categorical features (one-hot encoding)
    # and to pass through numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the yield prediction model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"Model R-squared score: {score:.4f}")

    # Save the trained model pipeline
    joblib.dump(model, model_path)
    print(f"Yield prediction model saved to {model_path}")

if __name__ == "__main__":
    train_yield_model()
