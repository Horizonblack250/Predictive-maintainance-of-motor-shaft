import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path of your dataset
data = pd.read_csv('0D.csv')

# Select the columns you want to use for modeling
# Here, we're using Vibration_1, Vibration_2, Vibration_3, Measured_RPM, and Vin
selected_columns = ['Vibration_1', 'Vibration_2', 'Vibration_3', 'Measured_RPM', 'V_in']
observations = data[selected_columns].values

# Normalize the data
scaler = StandardScaler()
observations = scaler.fit_transform(observations)

# Create an HMM model with an increased number of hidden states
num_states = 4  # Adjust the number of hidden states
model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=100, tol=1e-3)

# Train the HMM on your dataset
model.fit(observations)

# Perform inference using the Viterbi algorithm to find the most likely state sequence
state_sequence = model.predict(observations)

# You can also compute the likelihood of the observations given the model
log_likelihood = model.score(observations)

# Print the results
print("Most likely state sequence:", state_sequence)
print("Log likelihood of observations:", log_likelihood)
