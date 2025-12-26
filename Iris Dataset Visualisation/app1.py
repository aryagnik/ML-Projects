# import pandas as pd
# import streamlit as st
# import plotly.express as px

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

# df = pd.read_csv(url, header = None)

# print(df.head())

# df.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']

# st.title("Interactive Box Plot using Streamlit")

# selected_feature = st.selectbox(
#     'Select a feature: ',
#     ['SepalLength','SepalWidth','PetalLength','PetalWidth']
# )

# box_plot = px.box(df, x='Species', y=selected_feature)

# st.plotly_chart(box_plot)

import pandas as pd
import streamlit as st
import plotly.express as px

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
df = pd.read_csv(url, header=None)
df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# Page title
st.title("Iris Dataset Box Plot Analysis")

# Introductory text
st.markdown("""
    Welcome to the interactive Iris dataset box plot visualization app! 
    This tool allows you to explore the distribution of various features 
    across the three species of the famous Iris flower dataset.
""")

# Sidebar for feature selection
st.sidebar.header("Feature Selection")
selected_feature = st.sidebar.selectbox(
    'Choose a feature to visualize:', 
    ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
)

# Display selected feature in sidebar
st.sidebar.write(f"Selected Feature: **{selected_feature}**")

# Generate box plot using Plotly
fig = px.box(
    df, 
    x='Species', 
    y=selected_feature, 
    color='Species',
    points="all",  # Shows all points on the plot
    template="plotly_dark",  # Change the theme
    title=f"Distribution of {selected_feature} Across Iris Species"
)

# Add hover information and customize appearance
fig.update_traces(marker=dict(size=10, opacity=0.7), hoverinfo="all")

# Display the box plot in the main layout
st.plotly_chart(fig, use_container_width=True)

# Additional explanation
st.markdown("""
    #### About the Iris Dataset:
    The Iris dataset is a famous dataset in the world of machine learning and data science. It includes 150 observations of iris flowers, classified into three species:
    
    - Setosa
    - Versicolor
    - Virginica
    
    The features recorded for each flower include:
    
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    
    The box plot helps in visualizing the distribution and spread of these features, providing insights into how they vary across species.
""")
