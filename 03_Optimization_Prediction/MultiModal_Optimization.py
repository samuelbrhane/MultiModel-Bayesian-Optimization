import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from scipy.spatial import distance

print(f"Script started at {datetime.now()}")
start = datetime.now()

# Device and precision settings
device = torch.device("cpu")
dtype = torch.double
torch.manual_seed(3)

# Constants
new_candidates = 15
ref_point = [550, 10]  # Reference point for qNEHVI [activity, stability]
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]
properties = ["Overpotential", "Overpotential change"]
std = ["Overpotential_std", "Overpotential change_std"]
weight_fixed = 0.5
weight_multi = 0.5

# Directories
current_dir = os.getcwd()
optimization_dir = os.path.join(current_dir, "03_Optimization_Prediction")
optimization_output_dir = os.path.join(optimization_dir, "Optimization_Output")
path_data = os.path.join(optimization_dir, "HT_Data")
composition_space_path = os.path.join(optimization_dir, "Composition_space_10%.xlsx")

# Ensure output directories exist
os.makedirs(optimization_output_dir, exist_ok=True)

# Load data
comp_space_total = pd.read_excel(composition_space_path)
files = [f for f in os.listdir(path_data) if f.endswith(".xlsx")]
HT_data_import = pd.read_excel(os.path.join(path_data, files[0]))

# Normalize compositions
HT_data_import[elements] = HT_data_import[elements].div(100, axis=0)
HT_data_import[properties] = -HT_data_import[properties]
ref_point = (pd.Series(ref_point) * -1).tolist()

# Select relevant columns
HT_data = HT_data_import[elements + properties + std]
HT_data.loc[:, std] = HT_data[std].fillna(0)

# Remove existing compositions from total space
index_to_drop = []
for index, row in HT_data[elements].iterrows():
    index_to_drop.append(int(np.where((np.array(comp_space_total / 100) == np.array(row)).all(axis=1))[0]))

comp_space_reduced = comp_space_total.drop(index=index_to_drop).reset_index(drop=True)

# Define inputs for models
x_init = torch.tensor(np.array(HT_data[elements]), dtype=dtype)
x_init_multitask = torch.cat([
    torch.cat([x_init, torch.zeros(x_init.size(0), 1)], dim=-1),
    torch.cat([x_init, torch.ones(x_init.size(0), 1)], dim=-1),
])

# Standardize outputs
y_init = HT_data[properties].values
scaler = preprocessing.StandardScaler().fit(y_init)
y_init_scaled = scaler.transform(y_init)
y_init_scaled = torch.tensor(y_init_scaled, dtype=dtype)

y_init_multitask = torch.cat([y_init_scaled[:, 0], y_init_scaled[:, 1]], dim=0).unsqueeze(-1)
y_init_std = HT_data[std].values / scaler.scale_
y_init_std = torch.tensor(y_init_std, dtype=dtype)
y_init_std_multitask = torch.cat([y_init_std[:, 0], y_init_std[:, 1]], dim=0).unsqueeze(-1)

# Train FixedNoiseMultiTaskGP
model_fixed = FixedNoiseMultiTaskGP(x_init_multitask, y_init_multitask, train_Yvar=y_init_std_multitask, task_feature=-1)
mll_fixed = ExactMarginalLogLikelihood(model_fixed.likelihood, model_fixed)
fit_gpytorch_model(mll_fixed)

# Train MultiTaskGP
model_multi = MultiTaskGP(x_init_multitask, y_init_multitask, task_feature=-1)
mll_multi = ExactMarginalLogLikelihood(model_multi.likelihood, model_multi)
fit_gpytorch_model(mll_multi)

# Predict using FixedNoiseMultiTaskGP and MultiTaskGP
x_task0 = torch.cat([x_init, torch.zeros(x_init.size(0), 1)], dim=-1)
x_task1 = torch.cat([x_init, torch.ones(x_init.size(0), 1)], dim=-1)

with torch.no_grad():
    # Predictions for FixedNoiseMultiTaskGP
    noise_task0 = torch.zeros_like(y_init_multitask[:x_task0.size(0)])
    noise_task1 = torch.zeros_like(y_init_multitask[:x_task1.size(0)])
    mean_fixed_task0 = model_fixed.likelihood(model_fixed(x_task0), noise=noise_task0).mean
    mean_fixed_task1 = model_fixed.likelihood(model_fixed(x_task1), noise=noise_task1).mean

    # Predictions for MultiTaskGP
    mean_multi_task0 = model_multi.likelihood(model_multi(x_task0)).mean
    mean_multi_task1 = model_multi.likelihood(model_multi(x_task1)).mean

# Combine results
mean_fixed = torch.cat([mean_fixed_task0, mean_fixed_task1], dim=0)
mean_multi = torch.cat([mean_multi_task0, mean_multi_task1], dim=0)

# Combine predictions for multitask output
mean_combined_task0 = weight_fixed * mean_fixed[:x_task0.size(0)] + weight_multi * mean_multi[:x_task0.size(0)]
mean_combined_task1 = weight_fixed * mean_fixed[x_task0.size(0):] + weight_multi * mean_multi[x_task0.size(0):]

# Stack the combined outputs into a single tensor (multitask format)
y_combined = torch.cat([mean_combined_task0, mean_combined_task1], dim=0).unsqueeze(-1)

x_combined = torch.cat([x_task0, x_task1], dim=0)


# Train a new FixedNoiseMultiTaskGP model using the combined predictions
combined_model = FixedNoiseMultiTaskGP(x_combined, y_combined, train_Yvar=y_init_std_multitask, task_feature=-1)
mll_combined = ExactMarginalLogLikelihood(combined_model.likelihood, combined_model)
fit_gpytorch_model(mll_combined)

# Define acquisition function using the newly trained combined model
ref_point_scaled = scaler.transform(np.array(ref_point).reshape(1, -1)).flatten().tolist()

acq_function = qNoisyExpectedHypervolumeImprovement(
    model=combined_model,
    ref_point=ref_point_scaled,
    X_baseline=x_init,
)

# Optimize acquisition function to suggest new candidates
bounds = torch.cat([torch.zeros(x_init.size(1), dtype=dtype), torch.ones(x_init.size(1), dtype=dtype)]).view(2, -1)
equality_constraint = [(torch.arange(len(elements), dtype=torch.long), torch.ones(len(elements), dtype=dtype), 1.0)]

candidates, _ = optimize_acqf(
    acq_function=acq_function,
    bounds=bounds,
    equality_constraints=equality_constraint,
    q=new_candidates,
    sequential=True,
    num_restarts=300,
    raw_samples=1024,
    options={"batch_limit": 5, "maxiter": 400},
)

# Scale candidates back to the original composition space
candidates = candidates.detach().numpy() * 100

# Save raw candidates (Original decimal values)
candidates_df = pd.DataFrame(candidates, columns=elements)

# Add predictions from both models and rescale them to original values
with torch.no_grad():
    # Predict activity and stability for new candidates using both models
    candidates_tensor = torch.tensor(candidates / 100, dtype=dtype)
    
    candidates_fixed_task0 = torch.cat([candidates_tensor, torch.zeros(candidates_tensor.size(0), 1)], dim=-1)
    candidates_fixed_task1 = torch.cat([candidates_tensor, torch.ones(candidates_tensor.size(0), 1)], dim=-1)
    
    # Predictions from the FixedNoiseMultiTaskGP model
    fixed_predictions_task0 = model_fixed.likelihood(model_fixed(candidates_fixed_task0)).mean.squeeze(-1).numpy()
    fixed_predictions_task1 = model_fixed.likelihood(model_fixed(candidates_fixed_task1)).mean.squeeze(-1).numpy()
    
    # Predictions from the MultiTaskGP model
    multi_predictions_task0 = model_multi.likelihood(model_multi(candidates_fixed_task0)).mean.squeeze(-1).numpy()
    multi_predictions_task1 = model_multi.likelihood(model_multi(candidates_fixed_task1)).mean.squeeze(-1).numpy()

# Rescale the predictions back to the original scale
fixed_predictions_task0_original = -scaler.inverse_transform(
    np.column_stack([fixed_predictions_task0, np.zeros_like(fixed_predictions_task0)]))[:, 0]
fixed_predictions_task1_original = -scaler.inverse_transform(
    np.column_stack([np.zeros_like(fixed_predictions_task1), fixed_predictions_task1]))[:, 1]

multi_predictions_task0_original = -scaler.inverse_transform(
    np.column_stack([multi_predictions_task0, np.zeros_like(multi_predictions_task0)]))[:, 0]
multi_predictions_task1_original = -scaler.inverse_transform(
    np.column_stack([np.zeros_like(multi_predictions_task1), multi_predictions_task1]))[:, 1]


# Append rescaled predictions to candidates_df
candidates_df["FixedNoiseModel_Activity"] = fixed_predictions_task0_original
candidates_df["FixedNoiseModel_Stability"] = fixed_predictions_task1_original
candidates_df["MultiTaskModel_Activity"] = multi_predictions_task0_original
candidates_df["MultiTaskModel_Stability"] = multi_predictions_task1_original

# Save raw candidates with rescaled predictions
now = datetime.now()
candidates_df.to_excel(
    os.path.join(optimization_output_dir, f"Next_candidates_combined_original_{now.strftime('%Y%m%d_%H-%M-%S')}.xlsx"),
    index=False,
)

# Match candidates to the closest compositions in the reduced composition space
candidates_selection = pd.DataFrame(columns=elements + ["FixedNoiseModel_Activity", "FixedNoiseModel_Stability", "MultiTaskModel_Activity", "MultiTaskModel_Stability"],
                                    index=[*range(0, new_candidates, 1)], dtype="float64")

for index, i in enumerate(candidates):
    dist = [distance.euclidean(i, j) for j in np.array(comp_space_reduced)]
    index_closest_composition = np.argmin(dist)
    closest_composition = comp_space_reduced.iloc[index_closest_composition]
    comp_space_reduced = comp_space_reduced.drop(index=index_closest_composition).reset_index(drop=True)
    
    # Add predictions for the closest composition
    candidates_selection.iloc[index, :len(elements)] = closest_composition
    candidates_selection.iloc[index, len(elements):] = candidates_df.iloc[index, len(elements):]

# Reorder candidates to prioritize Co max
Co_max_index = candidates_selection["Co"].idxmax()
new_idx = [Co_max_index] + [i for i in range(len(candidates_selection)) if i != Co_max_index]
candidates_selection = candidates_selection.iloc[new_idx].reset_index(drop=True)

# Save rounded and matched candidates (Integer values)
candidates_selection.to_excel(
    os.path.join(optimization_output_dir, f"Next_candidates_combined_{now.strftime('%Y%m%d_%H-%M-%S')}.xlsx"),
    index=False,
)

# Print script execution time and completion message
end = datetime.now()
print(f"Script ended at {datetime.now()}")
print(f"Script execution took {end - start}")
