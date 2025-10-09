# Description: A performant, responsive plot using the Lato font, with an updated info panel.

import pandas as pd
import plotly.express as px
import os

# --- Get the directory where the script is located ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()  # Fallback for environments like Jupyter

# --- 1. Load Data Processed by R ---
try:
    final_df = pd.read_csv("pca_results.csv")
    variance_df = pd.read_csv("pca_variance.csv")
except FileNotFoundError as e:
    print(f"Error: Could not find '{e.filename}'.")
    print("Please run the 'perform_pca.R' script first.")
    exit()

var_pc1 = variance_df['var_PC1'].iloc[0]
var_pc2 = variance_df['var_PC2'].iloc[0]
var_pc3 = variance_df['var_PC3'].iloc[0]


# --- 2. Create the 3D PCA Plot ---
symbol_map = {'male': 'circle', 'female': 'x'}
color_map = {
    "Calliphora vicina": "#808080", "Chrysomya albiceps normal": "#F5DEB3",
    "Chrysomya albiceps mutant": "#DA70D6", "Chrysomya bezziana": "#A52A2A",
    "Chrysomya megacephala": "#FFFF00", "Chrysomya rufifacies": "#8A2BE2",
    "Lucilia sericata": "#00FFFF", "Synthesiomyia nudiseta": "#FFB6C1"
}

fig = px.scatter_3d(
    final_df, x='PC1', y='PC2', z='PC3',
    color='species', symbol='sex', symbol_map=symbol_map, color_discrete_map=color_map,
    title="3D PCA of Fly Wing Harmonics (Hover to Highlight, Click for Details)",
    labels={
        'PC1': f'PC1 ({var_pc1:.1%})',
        'PC2': f'PC2 ({var_pc2:.1%})',
        'PC3': f'PC3 ({var_pc3:.1%})'
    },
    custom_data=['species', 'sex']
)

# --- 3. Customize traces and legend behavior ---
for trace in fig.data:
    if 'female' not in trace.name:
        species_name = trace.name.split(',')[0]
        trace.legendgroup = species_name
        trace.name = f"<i>{species_name}</i>"
        trace.showlegend = True
        trace.marker.update(size=6, line=dict(width=0.5, color='Black'), opacity=0.8)
for trace in fig.data:
    if 'female' in trace.name:
        species_name = trace.name.split(',')[0]
        trace.legendgroup = species_name
        trace.showlegend = False
        trace.marker.update(size=4, line=dict(width=0.5), opacity=0.8)


# --- 4. Layout adjustments ---
fig.update_layout(
    hovermode=False,
    # ✅ Set the global font for the plot
    font=dict(family="Lato"),
    legend=dict(
        title_text="<b>Species</b><br><span style='font-size: 18px;'>(Male ●, Female ×)</span>",
        title_font=dict(size=25),
        font=dict(size=22),
        itemsizing='constant',
        bgcolor='rgba(255,255,255,0.7)',
    ),
    scene=dict(
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        zaxis_title_font=dict(size=16),
    ),
    title_font=dict(size=22),
)


# --- 5. Add interactivity via JavaScript ---
js_code = """
var plot_div = document.getElementsByClassName('plotly-graph-div')[0];

// --- 1. Create a custom HTML info panel and its style ---
var style = document.createElement('style');
style.innerHTML = `
/* Make the plot container a positioning context */
.plotly-graph-div {
    position: relative !important;
}
#info-panel {
    position: absolute;
    bottom: 20px;
    right: 20px;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.85);
    border: 1px solid #ccc;
    border-radius: 5px;
    /* ✅ Update font for the panel */
    font-family: Lato;
    font-size: 20px;
    pointer-events: none;
    z-index: 100;
}
`;
document.head.appendChild(style);

// Create the info panel div itself
var infoPanel = document.createElement('div');
infoPanel.id = 'info-panel';
infoPanel.innerHTML = 'Click on a point for details';
plot_div.appendChild(infoPanel);


// --- 2. Logic for Hover (Fading effect) ---
plot_div.on('plotly_hover', function(data){
    var point = data.points[0];
    var speciesName = point.data.legendgroup || point.data.name.split(",")[0];
    
    plot_div.data.forEach(function(trace, i) {
        var traceSpecies = trace.legendgroup || trace.name.split(",")[0];
        var newOpacity = (traceSpecies === speciesName) ? 0.9 : 0.05;
        if (trace.marker.opacity !== newOpacity) {
            Plotly.restyle(plot_div, {'marker.opacity': [newOpacity]}, [i]);
        }
    });
});

// --- 3. Logic to Reset Hover ---
plot_div.on('plotly_unhover', function(data){
    plot_div.data.forEach(function(trace, i) {
        if (trace.marker.opacity !== 0.8) {
            Plotly.restyle(plot_div, {'marker.opacity': [0.8]}, [i]);
        }
    });
});

// --- 4. Logic for Click (Update the custom HTML panel) ---
plot_div.on('plotly_click', function(data){
    var point = data.points[0];
    
    var label_x = plot_div.layout.scene.xaxis.title.text;
    var label_y = plot_div.layout.scene.yaxis.title.text;
    var label_z = plot_div.layout.scene.zaxis.title.text;

    // ✅ Capitalize the sex for display
    var sex = point.customdata[1];
    var capitalizedSex = sex.charAt(0).toUpperCase() + sex.slice(1);

    // ✅ Update text format: italic species, no "Sex:" label
    var click_text = `<b><i>${point.customdata[0]}</i></b><br>` +
                     `${capitalizedSex}<br>` +
                     `${label_x.split(' ')[0]}: ${point.x.toFixed(2)}<br>` +
                     `${label_y.split(' ')[0]}: ${point.y.toFixed(2)}<br>` +
                     `${label_z.split(' ')[0]}: ${point.z.toFixed(2)}`;
    
    infoPanel.innerHTML = click_text;
});
"""

# --- 6. Save the interactive plot ---
output_filename = "interactive_pca_plot.html"
output_path = os.path.join(script_dir, output_filename)
fig.write_html(output_path, post_script=js_code, include_plotlyjs=True)

print(f"✅ Successfully saved the interactive plot to: {output_path}")
