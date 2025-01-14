import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
import os

# Device and precision settings
device = torch.device("cpu")
dtype = torch.double
torch.manual_seed(3)

# Constants
elements = ["Co", "Fe", "Ni"]
properties = ["Activity Cycle 52", "Stability"]
objective_weights = {"activity": 0.7, "stability": 0.3} 
model_weights = {"gp": 0.5, "nn": 0.5}  

# Directories
current_dir = os.getcwd()
optimization_dir = os.path.join(current_dir, "03_Optimization_Prediction")
optimization_output_dir = os.path.join(optimization_dir, "Optimization_Output")
path_data = os.path.join(optimization_dir, "Data")
composition_space_path = os.path.join(optimization_dir, "5_percent_compositions.xlsx") 

# Ensure output directories exist
os.makedirs(optimization_output_dir, exist_ok=True)

# Load data
comp_space_total = pd.read_excel(composition_space_path)
files = [f for f in os.listdir(path_data) if f.endswith(".xlsx")]
HT_data_import = pd.read_excel(os.path.join(path_data, files[0]))

# Normalize compositions
HT_data_import[elements] = HT_data_import[elements].div(100, axis=0)

# Negate Activity for minimization
HT_data_import["Activity Cycle 52"] = -HT_data_import["Activity Cycle 52"]

# Select relevant columns
HT_data = HT_data_import[elements + properties]

# Prepare training data
x_train = torch.tensor(HT_data[elements].values, dtype=dtype)
y_train = torch.tensor(HT_data[properties].values, dtype=dtype)

# Normalize outputs for objective function
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.numpy())

# Combine Activity and Stability into a single objective
objective_values = objective_weights["activity"] * (-y_train_scaled[:, 0]) + objective_weights["stability"] * y_train_scaled[:, 1]
objective_values = torch.tensor(objective_values, dtype=dtype).unsqueeze(-1)

# Train Gaussian Process model
gp_model = SingleTaskGP(x_train, objective_values)
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_model(mll)

# Define batch acquisition function for Bayesian Optimization
best_f = objective_values.max()
acq_function = qExpectedImprovement(model=gp_model, best_f=best_f)

# Normalize composition space
comp_space_normalized = comp_space_total[elements].div(100, axis=0).values
bounds = torch.tensor([[0.0] * len(elements), [1.0] * len(elements)], dtype=dtype)

# Optimize acquisition function to suggest multiple candidates
gp_candidates, _ = optimize_acqf(
    acq_function=acq_function,
    bounds=bounds,
    q=10,  # Suggest 10 candidates
    num_restarts=20,
    raw_samples=512,
    options={"batch_limit": 5, "maxiter": 200},
)

# Rescale GP candidates back to original composition
gp_candidates = gp_candidates.detach().numpy() * 100

# Predict GP Activity and Stability for the entire composition space
gp_predictions = gp_model.posterior(torch.tensor(comp_space_normalized, dtype=dtype)).mean.detach().numpy()
comp_space_total["GP_Activity"] = -gp_predictions[:, 0]  # Negate Activity back to original scale
comp_space_total["GP_Stability"] = scaler.inverse_transform(
    np.column_stack([np.zeros_like(gp_predictions[:, 0]), gp_predictions[:, 1]])
)[:, 1]
print(comp_space_total)