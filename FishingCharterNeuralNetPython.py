import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Set the working directory
import os

# Read the CSV file into a DataFrame
fishingCharter = pd.read_csv("FishingCharter.csv")

# Display the DataFrame
print(fishingCharter)

# Display the DataFrame structure
print(fishingCharter.info())

# Display the DataFrame summary
print(fishingCharter.describe())

# Scale the AnnualIncome and CatchRate variables
scaler = MinMaxScaler()
fishingCharter[['AnnualIncomeScaled', 'CatchRateScaled']] = scaler.fit_transform(fishingCharter[['AnnualIncome', 'CatchRate']])

# Randomly split the dataset into training (75%) and testing (25%)
np.random.seed(591)
sampleSet = np.random.choice(fishingCharter.index, size=int(len(fishingCharter) * 0.75), replace=False)
fishingCharterTraining = fishingCharter.loc[sampleSet]
fishingCharterTesting = fishingCharter.loc[~fishingCharter.index.isin(sampleSet)]

# Define the neural network model
model = Sequential()
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare the training data
X_train = fishingCharterTraining[['AnnualIncomeScaled', 'CatchRateScaled']]
y_train = fishingCharterTraining['CharteredBoat']

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Use the trained model to make predictions on the testing data
X_test = fishingCharterTesting[['AnnualIncomeScaled', 'CatchRateScaled']]
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Display the probability predictions
print(y_pred_prob)

# Display the 0/1 predictions
print(y_pred)

# Evaluate the model using a confusion matrix
confusionMatrix = confusion_matrix(fishingCharterTesting['CharteredBoat'], y_pred)
print(confusionMatrix)

# Calculate and display the predictive accuracy
accuracy = accuracy_score(fishingCharterTesting['CharteredBoat'], y_pred)
print(accuracy)
