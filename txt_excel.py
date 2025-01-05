import pandas as pd

# File paths
input_txt_path = "Experiment_data_iteration3.txt"  
output_excel_path = "Experiment_data_iteration3.xlsx" 

# Read the text file into a DataFrame
data = pd.read_csv(input_txt_path, sep="\t")

# Save the DataFrame to an Excel file
data.to_excel(output_excel_path, index=False)

print(f"Data successfully converted to {output_excel_path}")
