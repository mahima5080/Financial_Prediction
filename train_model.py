import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load the Dataset
file_path = 'Financial Sample.xlsx'
df = pd.read_excel(file_path)

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Handle missing values
df['Discount Band'] = df['Discount Band'].fillna('None')

# 2. Features and Target Selection
categorical_cols = ['Segment', 'Country', 'Product', 'Discount Band']
numerical_cols = ['Units Sold', 'Manufacturing Price', 'Sale Price']
target_col = 'Sales'

X = df[categorical_cols + numerical_cols]
y = df[target_col]

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ]
)

X_processed = preprocessor.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 4. Build TensorFlow Neural Network
model = keras.Sequential([
    layers.Input(shape=(X_processed.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error']
)

print('Training TensorFlow model...')
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=0,
)
print('Model training complete!')

# 5. Generate Predictions
y_pred = model.predict(X_processed).flatten()

df['Predicted_Sales'] = np.round(y_pred, 2)
df['Prediction_Error'] = np.round(df['Sales'] - df['Predicted_Sales'], 2)
df['Absolute_Error'] = np.abs(df['Prediction_Error'])

# 6. Export to CSV for Power BI
output_csv = 'Financial_Predictions_PowerBI.csv'
df.to_csv(output_csv, index=False)
print(f"File saved successfully as '{output_csv}'!")