import pandas as pd
import numpy as np

data = pd.read_excel("Next_candidates_with_experimental.xlsx")

# Calculate squared errors
data['FixedNoiseModel_SquaredError_Activity'] = (data['FixedNoiseModel_Activity'] - data['Experimental_Activity'])**2
data['FixedNoiseModel_SquaredError_Stability'] = (data['FixedNoiseModel_Stability'] - data['Experimental_Stability'])**2

data['MultiTaskModel_SquaredError_Activity'] = (data['MultiTaskModel_Activity'] - data['Experimental_Activity'])**2
data['MultiTaskModel_SquaredError_Stability'] = (data['MultiTaskModel_Stability'] - data['Experimental_Stability'])**2

# Total squared errors for each model
total_squared_error_fixed = data['FixedNoiseModel_SquaredError_Activity'].sum() + data['FixedNoiseModel_SquaredError_Stability'].sum()
total_squared_error_multi = data['MultiTaskModel_SquaredError_Activity'].sum() + data['MultiTaskModel_SquaredError_Stability'].sum()

# Calculate weights inversely proportional to squared errors
weight_fixed = (1 / total_squared_error_fixed) / ((1 / total_squared_error_fixed) + (1 / total_squared_error_multi))
weight_multi = (1 / total_squared_error_multi) / ((1 / total_squared_error_fixed) + (1 / total_squared_error_multi))

print(f"Weight for FixedNoiseModel: {weight_fixed:.4f}")
print(f"Weight for MultiTaskModel: {weight_multi:.4f}")
