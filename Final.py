import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('0D.csv')

selected_columns = ['Vibration_1', 'Vibration_2', 'Vibration_3', 'Measured_RPM', 'V_in']
observations = data[selected_columns].values

scaler = StandardScaler()
observations = scaler.fit_transform(observations)

num_states = 4  
model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=100, tol=1e-3)

model.fit(observations)

state_sequence = model.predict(observations)

log_likelihood = model.score(observations)

print("Most likely state sequence:", state_sequence)
print("Log likelihood of observations:", log_likelihood)
