import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Get the directory where the script is located ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback for environments like Jupyter

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv(r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check the path.")
    exit()

if df['gender'].nunique() < 2:
    print("Error: The 'gender' column must contain at least two unique groups to perform LDA.")
    exit()

harmonic_columns = [f"{char}{i}" for char in 'abcd' for i in range(1, 11)]
X = df[harmonic_columns]
y = df['gender']

# --- 2. Perform LDA by Gender ---
lda = LinearDiscriminantAnalysis(n_components=1)
X_scaled = StandardScaler().fit_transform(X)
lda_results = lda.fit_transform(X_scaled, y)

final_df = pd.DataFrame({
    'LD1': lda_results.flatten(),
    'gender': df['gender'],
    'species': df['species']
})

print("LDA by gender complete. Generating KDE visualization...")

# --- 3. Visualize with Seaborn Kernel Density Plot ---

# --- START OF MODIFIED SECTION ---

# Define your custom color map
color_map = {
    "Calliphora vicina": "#808080", 
    "Chrysomya albiceps_normal": "#F5DEB3", 
    "Chrysomya albiceps_mutant": "#DA70D6",
    "Chrysomya bezziana": "#A52A2A", 
    "Chrysomya megacephala": "#FFFF00", 
    "Chrysomya rufifacies": "#8A2BE2", 
    "Lucilia sericata": "#00FFFF", 
    "Synthesiomyia nudiseta": "#FFB6C1"
}

# Create a figure and axes for the plot
plt.figure(figsize=(14, 8))

# Get unique species to loop through
species_list = final_df['species'].unique()

# Loop through each species to plot its densities with the correct color
for species in species_list:
    # Get the specific color for the current species from the map
    species_color = color_map.get(species)
    
    # Filter data for the current species
    species_df = final_df[final_df['species'] == species]
    
    # Plot male density with a dashed line and the species-specific color
    male_data = species_df[species_df['gender'] == 'male']
    if not male_data.empty:
        sns.kdeplot(data=male_data, x='LD1', color=species_color, linestyle='--', label=f'{species} (male)')

    # Plot female density with a solid line and the species-specific color
    female_data = species_df[species_df['gender'] == 'female']
    if not female_data.empty:
        sns.kdeplot(data=female_data, x='LD1', color=species_color, linestyle='-', label=f'{species} (female)')

# --- END OF MODIFIED SECTION ---

# --- 4. Customize and Save the Plot ---
plt.title('Kernel Density of Species Along the Axis of Gender Separation (LD1)', fontsize=16)
plt.xlabel('Linear Discriminant 1 (Maximizes Gender Separation)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Species & Gender')

# Save the plot to a file
output_filename = "species_density_on_lda_colored.png"
output_path = os.path.join(script_dir, output_filename)
plt.savefig(output_path)

print(f"Successfully saved the density plot to: {output_path}")