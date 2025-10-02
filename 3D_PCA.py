import pandas as pd
from sklearn.decomposition import PCA
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
    df = pd.read_csv(r"C:\Users\User\Documents\Bioinformatics_Year3_Sem2\Internship\Fly Project\normalized_efd_coefficients_10h.csv")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please check the path.")
    exit()

harmonic_columns = [f"{char}{i}" for char in 'abcd' for i in range(1, 11)]
X = df[harmonic_columns]

# --- 2. Perform PCA ---
pca = PCA(n_components=3)
X_scaled = StandardScaler().fit_transform(X)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2', 'PC3'])
final_df = pd.concat([df[['species', 'gender']].reset_index(drop=True), pca_df], axis=1)

# --- 3. Create the 3D Plot with Customizations ---
symbol_map = {'male': 'circle', 'female': 'x'}
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

fig = px.scatter_3d(
    final_df,
    x='PC1',
    y='PC2',
    z='PC3',
    color='species',
    symbol='gender',
    symbol_map=symbol_map,
    color_discrete_map=color_map,
    title="3D PCA of Fly Wing Harmonics",
    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'}
)

# --- START OF MODIFIED SECTION ---

# The global update_traces call is no longer needed for size and opacity
# fig.update_traces(marker=dict(size=6, opacity=0.8))

# Loop through each trace to selectively apply properties
for trace in fig.data:
    # Set opacity for all traces
    trace.marker.opacity = 0.8
    
    # Plotly names each trace based on its legend entries, e.g., "Species, gender"
    if 'female' in trace.name:
        # For female (x symbol), set a specific size
        trace.marker.size = 3
        trace.marker.line.width = 0.5 # Optional border for 'x'
    else:
        # For all others (male/circle), set a different size and the border
        trace.marker.size = 5
        trace.marker.line.width = 0.5
        trace.marker.line.color = 'Black'

# --- END OF MODIFIED SECTION ---

# --- 4. Save the Plot to an HTML File ---
output_filename = "interactive_pca_plot.html"
output_path = os.path.join(script_dir, output_filename)
fig.write_html(output_path)

print(f"Successfully saved the interactive plot to: {output_path}")