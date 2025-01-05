from datetime import datetime
import pandas as pd
import os
from sklearn.manifold import MDS
import numpy as np

print(f"Script started at {datetime.now()}")
start = datetime.now()

# Set working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Load experimental data and composition space
data_file = "Experiment_data.txt"
HT_data = pd.read_table(data_file)
comp_space = pd.read_table("Composition_space_10%.txt")

# Extract element compositions
elements = ["Co", "Mn", "Sb", "Sn", "Ti"]
HT_data_elements = HT_data[elements]
X = comp_space.to_numpy()

# Apply MDS to composition space for 2D visualization
mds = MDS(n_components=2, random_state=0, normalized_stress="auto")
X_transform_2d = mds.fit_transform(X)

# Add MDS coordinates to composition space and save
comp_space["MDS_X"] = X_transform_2d[:, 0]
comp_space["MDS_Y"] = X_transform_2d[:, 1]
comp_space.to_csv("Composition_space_MDS.txt", index=False, sep="\t")

# Match MDS coordinates to experimental data
index_comp = [
    int(np.where((np.array(comp_space[elements]) == np.array(row)).all(axis=1))[0])
    for _, row in HT_data_elements.iterrows()
]
HT_data["MDS_X"] = X_transform_2d[index_comp][:, 0]
HT_data["MDS_Y"] = X_transform_2d[index_comp][:, 1]

# Save updated experimental data
HT_data.to_csv(data_file.split(".")[0] + "_MDS.txt", index=False, sep="\t")

print(f"Script ended at {datetime.now()}")
end = datetime.now()
print(f"Script execution took {end - start}")
