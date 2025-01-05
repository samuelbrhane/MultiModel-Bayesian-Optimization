import os
import pandas as pd
import matplotlib.pyplot as plt


current_dir = os.getcwd() 

# Define Scaling directory as a subdirectory of the parent
scaling_dir = os.path.join(current_dir, "01_Multidimensional_Scaling")
scaling_dir = os.path.abspath(scaling_dir) 


# Define the Plot directory as another subdirectory of the parent
plot_dir = os.path.join(current_dir, "02_Visualization_Plots")
plot_dir = os.path.abspath(plot_dir)  

# Load data
experiment_data_path = os.path.join(scaling_dir, "Experiment_data_MDS.txt")
composition_space_path = os.path.join(scaling_dir, "Composition_space_MDS.txt")
experiment_data = pd.read_table(experiment_data_path)
composition_space = pd.read_table(composition_space_path)

# Get the latest iteration number for dynamic naming
current_iteration = max(experiment_data["Iteration"])

# Scatter plot of composition space and experimental data
scatter_plot_path = os.path.join(plot_dir, f"scatter_plot_iteration_{current_iteration}.png")
plt.figure(figsize=(10, 8))
plt.scatter(composition_space["MDS_X"], composition_space["MDS_Y"], c="lightgray", label="Full Composition Space", alpha=0.5)
scatter = plt.scatter(
    experiment_data["MDS_X"], experiment_data["MDS_Y"], c=experiment_data["Iteration"], cmap="viridis", label="Experimental Points", edgecolor="black"
)
plt.colorbar(scatter, label="Iteration")
plt.xlabel("MDS_X")
plt.ylabel("MDS_Y")
plt.title(f"Composition Space Exploration - Iteration {current_iteration}")
plt.legend()
plt.grid(True)
plt.savefig(scatter_plot_path)
plt.close()

# Pareto front plot
pareto_plot_path = os.path.join(plot_dir, f"pareto_plot_iteration_{current_iteration}.png")
plt.figure(figsize=(10, 6))
plt.scatter(experiment_data["Overpotential"], experiment_data["Overpotential change"], c="blue", label="Experimental Points", alpha=0.6)
pareto_mask = (experiment_data["Overpotential"] <= experiment_data["Overpotential"].min()) & (
    experiment_data["Overpotential change"] <= experiment_data["Overpotential change"].min()
)
pareto_points = experiment_data[pareto_mask]
plt.scatter(pareto_points["Overpotential"], pareto_points["Overpotential change"], c="red", label="Pareto Front Points", edgecolor="black")
plt.xlabel("Overpotential")
plt.ylabel("Overpotential Change")
plt.title(f"Pareto Front Visualization - Iteration {current_iteration}")
plt.legend()
plt.grid(True)
plt.savefig(pareto_plot_path)
plt.close()

# Print paths of saved plots
print(f"Scatter plot saved to: {scatter_plot_path}")
print(f"Pareto plot saved to: {pareto_plot_path}")
