import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the models and dataset
knn_model = joblib.load('trek_recommender_model.pkl')
scaler = joblib.load('scaler.pkl')
rf_cost_model = joblib.load('cost_prediction_model.pkl')
cost_scaler = joblib.load('cost_scaler.pkl')
dataset = pd.read_csv('Cleaned_Trekking_Dataset.csv')

# Title for the Streamlit app
st.title("Trek Recommendation and Cost Prediction System")

# Create tabs for different sections, including a homepage
tabs = st.tabs(["Home", "Trek Recommendations", "Trek Cost Prediction", "Trekking Data Visualizations"])

# ---------------------------------------------------
# HOMEPAGE (First Tab)
# ---------------------------------------------------

with tabs[0]:
    st.header("Welcome to the Trek Recommendation System!")
    st.write("""
        This web application helps you to plan your trekking adventures by recommending the best treks based on your preferences, predicting the cost of treks, and visualizing trends in trekking data.
        
        ### Features:
        - **Trek Recommendations**: Enter your preferences (e.g., budget, time, altitude) and get personalized trek recommendations.
        - **Cost Prediction**: Predict the estimated cost of your trekking trip based on input parameters such as duration, altitude, and group size.
        - **Data Visualizations**: Explore trends in trekking data, including costs, fitness levels, and best travel times.
        
        ### How to Use:
        1. Navigate to the **Trek Recommendations** tab to get recommendations.
        2. Use the **Cost Prediction** tab to estimate the cost of your trek.
        3. Explore the **Trekking Data Visualizations** tab to view various insights from trekking data.
        
        Enjoy your adventure planning!
    """)
    
    # Optionally, you could add an image related to trekking
    st.image("https://example.com/path-to-your-trekking-image.jpg", caption="Plan Your Next Trekking Adventure", use_column_width=True)

# ---------------------------------------------------
# TREK RECOMMENDATION SYSTEM (Second Tab)
# ---------------------------------------------------

with tabs[1]:
    st.header("Trek Recommendation System")
    cost = st.number_input("Enter your budget (Cost):", min_value=100, max_value=10000)
    time = st.number_input("Enter the trek duration (in days):", min_value=1, max_value=30)
    altitude = st.number_input("Enter the maximum altitude you are comfortable with (in meters):", min_value=1000, max_value=8000)
    fitness_level = st.slider("Select your fitness level:", 1, 5, key="fitness_slider_1")

    # Guide or No Guide input
    guide_choice = st.selectbox("Do you prefer a guide or no guide?", ("Guide", "No Guide"))
    guide_numeric = 1 if guide_choice == "Guide" else 0

    group_size = st.number_input("Enter the preferred group size:", min_value=1, max_value=100)

    if st.button('Get Trek Recommendations'):
        input_data = np.array([[cost, time, altitude, fitness_level, guide_numeric, group_size]])
        input_scaled = scaler.transform(input_data)
        distances, indices = knn_model.kneighbors(input_scaled)

        st.subheader("Top 5 Recommended Treks for You:")
        for idx in indices[0]:
            trek_info = dataset.iloc[idx]
            st.write(f"Trek: {trek_info['Trek_cleaned']}, Cost: {trek_info['Cost_cleaned']}, "
                     f"Time: {trek_info['Time_cleaned']} days, Max Altitude: {trek_info['Max_Altitude_cleaned']} meters, "
                     f"Group Size: {trek_info['trekkig_group_size']}, Guide: {trek_info['Guide_or_no_guide']}")

# ---------------------------------------------------
# TREK COST PREDICTION SYSTEM (Third Tab)
# ---------------------------------------------------

with tabs[2]:
    st.header("Trek Cost Prediction (Random Forest)")
    time_input = st.number_input("Enter trek duration (in days):", min_value=1, max_value=30)
    altitude_input = st.number_input("Enter maximum altitude (in meters):", min_value=1000, max_value=8000)
    fitness_level_input = st.slider("Select your fitness level:", 1, 5, key="fitness_slider_2")
    group_size_input = st.number_input("Enter group size:", min_value=1, max_value=100)

    if st.button('Predict Trek Cost'):
        input_data_cost = np.array([[time_input, altitude_input, fitness_level_input, group_size_input]])
        input_scaled_cost = cost_scaler.transform(input_data_cost)
        predicted_cost = rf_cost_model.predict(input_scaled_cost)
        st.subheader(f"Predicted Trek Cost: ${predicted_cost[0]:,.2f}")

# ---------------------------------------------------
# DATA VISUALIZATIONS (Fourth Tab)
# ---------------------------------------------------

with tabs[3]:
    st.header("Trekking Data Visualizations")

    # Bar chart: Average cost per fitness level
    fitness_cost_df = dataset.groupby('fitness_level_cleaned')['Cost_cleaned'].mean().reset_index()
    st.subheader("Average Trek Cost by Fitness Level")
    fig, ax = plt.subplots()
    sns.barplot(x='fitness_level_cleaned', y='Cost_cleaned', data=fitness_cost_df, ax=ax)
    ax.set_xlabel("Fitness Level")
    ax.set_ylabel("Average Cost")
    st.pyplot(fig)

    # Line graph: Trek cost over time
    st.subheader("Trek Costs Over Time")
    cost_time_df = dataset.groupby('year')['Cost_cleaned'].mean().reset_index()
    fig = px.line(cost_time_df, x='year', y='Cost_cleaned', title="Average Trek Cost Over the Years")
    st.plotly_chart(fig)

    # Scatter plot: Trek cost vs group size
    st.subheader("Trek Cost vs Group Size")
    fig, ax = plt.subplots()
    ax.scatter(dataset['trekkig_group_size'], dataset['Cost_cleaned'], alpha=0.5)
    ax.set_xlabel("Group Size")
    ax.set_ylabel("Trek Cost")
    st.pyplot(fig)

    # Plotly bar chart: Best travel time frequency
    st.subheader("Best Travel Time Frequency")
    best_travel_df = dataset['Best_travel_time_1'].value_counts().reset_index()
    best_travel_df.columns = ['Best Travel Time', 'Frequency']
    fig = px.bar(best_travel_df, x='Best Travel Time', y='Frequency', title="Best Time to Travel by Frequency")
    st.plotly_chart(fig)
