import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from botorch.optim import optimize_acqf
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from datetime import datetime

# Setup
print(f"Script started at {datetime.now()}")
start = datetime.now()

# Device and precision settings
device = torch.device("cpu")
dtype = torch.double
torch.manual_seed(42)

# Constants
elements = ["Co", "Fe", "Ni"]
properties = ["Activity Cycle 52", "Stability"]
objective_weights = {"activity": 1.0, "stability": -1.0}  # Define objective: Minimize Activity and Maximize Stability
new_candidates = 10  # Number of candidates to explore

# Directories
current_dir = os.getcwd()
optimization_dir = os.path.join(current_dir, "03_Optimization_Prediction")
optimization_output_dir = os.path.join(optimization_dir, "Optimization_Output")
path_data = os.path.join(optimization_dir, "Data")
composition_space_path = os.path.join(optimization_dir, "compositions.xlsx")

# Ensure output directories exist
os.makedirs(optimization_output_dir, exist_ok=True)

# Load data
comp_space_total = pd.read_excel(composition_space_path)
files = [f for f in os.listdir(path_data) if f.endswith(".xlsx")]
HT_data_import = pd.read_excel(os.path.join(path_data, files[0]))

# Normalize compositions
HT_data_import[elements] = HT_data_import[elements].div(100, axis=0)
HT_data_import["Activity Cycle 52"] = -HT_data_import["Activity Cycle 52"]  # Negate for minimization
HT_data = HT_data_import[elements + properties]

# Prepare training data
x_train = torch.tensor(HT_data[elements].values, dtype=dtype)
y_train = torch.tensor(HT_data[properties].values, dtype=dtype)

# Normalize outputs
scaler = MinMaxScaler()
y_train_scaled = torch.tensor(scaler.fit_transform(y_train.numpy()), dtype=dtype)

# Train Neural Network for Extrapolation
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: Activity and Stability
        )

    def forward(self, x):
        return self.layers(x)

model_nn = NeuralNetwork()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
criterion = nn.MSELoss()

x_train_nn = torch.tensor(HT_data[elements].values, dtype=torch.float32)
y_train_nn = torch.tensor(y_train.numpy(), dtype=torch.float32)

for epoch in range(2000):
    optimizer.zero_grad()
    predictions = model_nn(x_train_nn)
    loss = criterion(predictions, y_train_nn)
    loss.backward()
    optimizer.step()

# Predict for the composition space using NN
x_test_nn = torch.tensor(comp_space_total[elements].div(100, axis=0).values, dtype=torch.float32)
nn_predictions = model_nn(x_test_nn).detach().numpy()

# Train GP models for Activity and Stability
gp_activity = SingleTaskGP(x_train, (-y_train_scaled[:, 0]).unsqueeze(-1))
mll_activity = ExactMarginalLogLikelihood(gp_activity.likelihood, gp_activity)
fit_gpytorch_model(mll_activity)

gp_stability = SingleTaskGP(x_train, y_train_scaled[:, 1].unsqueeze(-1))
mll_stability = ExactMarginalLogLikelihood(gp_stability.likelihood, gp_stability)
fit_gpytorch_model(mll_stability)

# Define combined objective function
def combined_objective(nn_act, nn_stab, gp_act, gp_stab, w_gp=0.3, w_nn=0.7):
    combined_act = w_gp * gp_act + w_nn * nn_act
    combined_stab = w_gp * gp_stab + w_nn * nn_stab
    return objective_weights["activity"] * combined_act + objective_weights["stability"] * combined_stab

# Evaluate and rank all candidates
comp_space_total["NN_Activity"] = -nn_predictions[:, 0]
comp_space_total["NN_Stability"] = nn_predictions[:, 1]
comp_space_total["GP_Activity"] = -scaler.inverse_transform(
    np.column_stack([-gp_activity.posterior(x_test_nn).mean.detach().numpy().squeeze(),
                     np.zeros_like(nn_predictions[:, 1])]))[:, 0]
comp_space_total["GP_Stability"] = scaler.inverse_transform(
    np.column_stack([np.zeros_like(nn_predictions[:, 0]),
                     gp_stability.posterior(x_test_nn).mean.detach().numpy().squeeze()]))[:, 1]

comp_space_total["Combined_Score"] = combined_objective(
    comp_space_total["NN_Activity"],
    comp_space_total["NN_Stability"],
    comp_space_total["GP_Activity"],
    comp_space_total["GP_Stability"],
)

top_candidates = comp_space_total.sort_values(by="Combined_Score").head(new_candidates)

# Save top candidates
now = datetime.now()
top_candidates.to_excel(os.path.join(optimization_output_dir, f"Best_Candidates_{now.strftime('%Y%m%d_%H-%M-%S')}.xlsx"), index=False)

# Print completion
end = datetime.now()
print(f"Script ended at {datetime.now()}")
print(f"Script execution took {end - start}")
