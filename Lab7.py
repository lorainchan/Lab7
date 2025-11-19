import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# Configuration (Bonus: Layout)
st.set_page_config(
    page_title="Personalized Iris Dashboard",
    layout="wide", # Sets the page to use the full width of the screen
    initial_sidebar_state="expanded"
)

# Custom Colors(Bonus: Color Scheme)
IRIS_COLOR_MAP = {
    'Setosa': '#34A853',      # Green/Leaf color
    'Versicolor': '#4285F4',  # Blue/Sky color
    'Virginica': '#9B59B6'    # Purple/Flower color
}

# 1. Load Data
@st.cache_data
def load_iris_data():
    """Loads the Iris dataset from sklearn and converts it to a DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
    
    # Map numerical species index to actual names
    target_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    df['Species'] = df['Species'].map(target_names)
    
    return df

df = load_iris_data()

st.title("Iris Data Visualization")
st.markdown("Exploring the characteristics of three Iris species using a custom color scheme.")

# Add a personalized touch
st.balloons() 

# 2. Sidebar Filter (One filter)
st.sidebar.header("Filter Options")

selected_species = st.sidebar.selectbox(
    "Select Species",
    options=['All'] + list(df['Species'].unique())
)

if selected_species != 'All':
    filtered_df = df[df['Species'] == selected_species]
else:
    filtered_df = df

st.header(f"{selected_species} Species Data")

# 3. Data Summary Section (st.metric, st.dataframe, or st.table)
col_m1, col_m2, col_m3 = st.columns(3)

# Summary Metric
col_m1.metric("Total Observations", len(filtered_df))

# Average Sepal Length
avg_sepal = filtered_df['Sepal Length'].mean()
col_m2.metric("Avg. Sepal Length (cm)", f"{avg_sepal:.2f}")

# Average Petal Length
avg_petal = filtered_df['Petal Length'].mean()
col_m3.metric("Avg. Petal Length (cm)", f"{avg_petal:.2f}")

st.markdown("---")

# Layout for Visualizations (Bonus: Improved Layout)
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    # 4. First Visualization (Scattered plot)
    st.subheader("Petal Length vs. Petal Width (Scatter Plot)")
    
    fig_scatter = px.scatter(
        filtered_df, 
        x='Petal Length', 
        y='Petal Width', 
        color='Species',
        color_discrete_map=IRIS_COLOR_MAP, # Bonus: Custom Color Scheme
        title=f"Petal Dimensions for {selected_species} Species",
        template="simple_white" # Bonus: Personalized Chart Template
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with plot_col2:
    # 5. Second Visualization (Histogram)
    st.subheader("Distribution of Sepal Length (Histogram)")
    
    fig_hist = px.histogram(
        filtered_df, 
        x='Sepal Length', 
        color='Species',
        color_discrete_map=IRIS_COLOR_MAP, # Bonus: Custom Color Scheme
        marginal="box", 
        opacity=0.7, # Slightly lower opacity for better color blend
        title=f"Sepal Length Distribution for {selected_species} Species",
        template="simple_white" # Bonus: Personalized Chart Template
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# Display Raw Data (Required for summary section)
st.subheader("Filtered Raw Data")
st.dataframe(filtered_df, use_container_width=True)