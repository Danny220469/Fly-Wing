import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

# --- Get the directory where the script is located ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback for environments like Jupyter

# --- 1. Load and Prepare the Data ---
try:
    # Ensure the path to your CSV is correct
    df = pd.read_csv(r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check the path.")
    exit()

# --- Define X and y for SPECIES-ONLY LDA ---
harmonic_columns = [f"{char}{i}" for char in 'abcd' for i in range(1, 11)]
X = df[harmonic_columns]
y = df['species']  # The target for the LDA is now just the species

# --- 2. Perform LDA ---
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize LDA for 8 species groups (n_components will be at most 7)
lda = LinearDiscriminantAnalysis(n_components=3) 
ld_components = lda.fit_transform(X_scaled, y)

# Create a DataFrame for plotting
lda_df = pd.DataFrame(data=ld_components, columns=['LD1', 'LD2', 'LD3'])
# Combine with original species and gender for plotting
final_df = pd.concat([df[['species', 'gender']].reset_index(drop=True), lda_df], axis=1)

# --- 3. Create the 3D Plot with Customizations ---

# Define a symbol map for gender
symbol_map = {'male': 'circle', 'female': 'x'}

# Define a custom color map with full species names
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

# Create the figure
fig = px.scatter_3d(
    final_df,
    x='LD1',
    y='LD2',
    z='LD3',
    color='species',          # Color by species
    symbol='gender',          # Still show gender with symbols
    hover_name='species',
    symbol_map=symbol_map,
    color_discrete_map=color_map,
    title="3D LDA of Fly Wing Harmonics (by Species)",
    labels={'LD1': f'LD1 ({lda.explained_variance_ratio_[0]:.1%})', 
            'LD2': f'LD2 ({lda.explained_variance_ratio_[1]:.1%})', 
            'LD3': f'LD3 ({lda.explained_variance_ratio_[2]:.1%})'}
)

# --- START OF MODIFIED SECTION ---

# Loop through each trace to selectively apply properties
for trace in fig.data:
    # Set opacity for all traces
    trace.marker.opacity = 0.8
    
    # Plotly names each trace based on its legend entries, e.g., "Species, gender"
    if 'female' in trace.name:
        # For female (square symbol), set a specific size
        trace.marker.size = 3
        trace.marker.line = dict(width=0.5, color='Black')
    else:
        # For all others (male/circle), set a different size
        trace.marker.size = 5
        trace.marker.line = dict(width=0.5, color='Black')

# --- END OF MODIFIED SECTION ---

# --- 4. Save the Plot to an HTML File ---
output_filename = "interactive_lda_plot_species_only.html"
output_path = os.path.join(script_dir, output_filename)
fig.write_html(output_path)

print(f"Successfully saved the interactive plot to: {output_path}")