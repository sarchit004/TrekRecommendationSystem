# Import necessary libraries
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

# Function to create a star rating with yellow stars
def display_stars(rating):
    full_star = '<span style="color:gold;">★</span>'
    empty_star = '<span style="color:gray;">☆</span>'
    stars = full_star * int(rating) + empty_star * (5 - int(rating))
    return stars

# Set wide layout for better screen space usage
# st.set_page_config(layout="wide")

# Helper function to load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load external CSS file
local_css("style.css")

# Load the image as the background
def get_base64_of_bin_file(image_file):
    with open(image_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_of_bin_file("BannerImage.png")

# Injecting custom HTML and CSS for the banner with the uploaded image
st.markdown(
    f"""
    <style>
    .banner {{
        position: relative;
        height: 400px;
        width: 100%;
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .banner h1 {{
        color: white;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }}
    </style>
    <div class="banner">
        <h1>Trek Recommendation and Cost Prediction System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the models and dataset
knn_model = joblib.load('trek_recommender_model.pkl')  # Correct model path for KNN recommendation
scaler = joblib.load('trek_recommend_scaler.pkl')      # Correct scaler for KNN model
rf_cost_model = joblib.load('cost_prediction_model.pkl')  # Correct model path for cost prediction
cost_scaler = joblib.load('cost_scaler.pkl')           # Correct scaler for cost prediction
dataset = pd.read_csv('NoteBook/CleanedTrekDataset.csv')

# Create tabs for different sections, including a homepage
tabs = st.tabs(["Home", "Trek Recommendations", "Trek Cost Prediction", "Trekking Data Visualizations"])

# ---------------------------------------------------
# HOMEPAGE (First Tab)
# ---------------------------------------------------

with tabs[0]:
    st.header("Welcome to the Trek Recommendation System!")
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-intro">
    This web application helps you to plan your trekking adventures by recommending the best treks based on your preferences, predicting the cost of treks, and visualizing trends in trekking data.
    </div>
    
    ### Features:
    - **Trek Recommendations**: Enter your preferences (e.g., budget, time, altitude) and get personalized trek recommendations.
    - **Cost Prediction**: Predict the estimated cost of your trekking trip based on input parameters such as duration, altitude, and group size.
    - **Data Visualizations**: Explore trends in trekking data, including costs, fitness levels, and best travel times.
    
    ### How to Use:
    1. Navigate to the **Trek Recommendations** tab to get recommendations.
    2. Use the **Cost Prediction** tab to estimate the cost of your trek.
    3. Explore the **Trekking Data Visualizations** tab to view various insights from trekking data.
    
    <div class="custom-intro">
    Enjoy your adventure planning!
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# TREK RECOMMENDATION SYSTEM (Second Tab)
# ---------------------------------------------------

with tabs[1]:
    st.header("Trek Recommendation System")
    # st.markdown('<div class="card">', unsafe_allow_html=True)

    # Two-column layout for the input form
    col1, col2 = st.columns(2)

    with col1:
        cost = st.number_input("Enter your budget (Cost):", min_value=100, max_value=10000)
        time = st.number_input("Enter the trek duration (in days):", min_value=1, max_value=30)
        altitude = st.number_input("Enter the maximum altitude you are comfortable with (in meters):", min_value=1000, max_value=8000)

    with col2:
        fitness_level = st.slider("Select your fitness level:", 1, 5, key="fitness_slider_1")
        guide_choice = st.selectbox("Do you prefer a guide or no guide?", ("Guide", "No Guide"))
        guide_numeric = 1 if guide_choice == "Guide" else 0
        group_size = st.number_input("Enter the preferred group size:", min_value=1, max_value=100)

    if st.button('Get Trek Recommendations', key="recommend_button"):
        input_data = np.array([[cost, time, altitude, fitness_level, guide_numeric, group_size]])
        input_scaled = scaler.transform(input_data)  # Use the correct scaler

        # Get the nearest neighbors from the KNN model
        distances, indices = knn_model.kneighbors(input_scaled)

        # Apply custom style for cards
        st.markdown("""
            <style>
            .trek-card {
                background-color: #ffffff;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                transition: box-shadow 0.3s ease-in-out;
            }

            .trek-card:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }

            .trek-card h4 {
                color: #1abc9c;
                margin: 0 0 10px 0;
            }

            .trek-card p {
                color: #34495e;
                margin: 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display trek recommendations as cards with yellow star ratings
        for idx in indices[0]:
            trek_info = dataset.iloc[idx]
            trek_rating = trek_info['Rating']  # Assuming the column for rating is 'Rating'
            trek_stars = display_stars(trek_rating)  # Convert rating to yellow stars

            st.markdown(f"""
            <div class="trek-card">
                <h4>{trek_info['Trek']}</h4>
                <p><strong>Cost:</strong> {trek_info['Cost']} USD</p>
                <p><strong>Time:</strong> {trek_info['Time']} days</p>
                <p><strong>Max Altitude:</strong> {trek_info['Max_Altitude']} meters</p>
                <p><strong>Group Size:</strong> {trek_info['Trekking_GroupSize']}</p>
                <p><strong>Guide:</strong> {'Guide' if trek_info['Guide_or_no_guide'] == 1 else 'No Guide'}</p>
                <p><strong>Rating:</strong> {trek_stars}</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------
# TREK COST PREDICTION SYSTEM (Third Tab)
# ---------------------------------------------------

with tabs[2]:
    st.header("Trek Cost Prediction (Random Forest)")
    # st.markdown('<div class="stCard">', unsafe_allow_html=True)

    # Two-column layout for the cost prediction form
    col1, col2 = st.columns(2)

    with col1:
        time_input = st.number_input("Enter trek duration (in days):", min_value=1, max_value=30, key="cost_duration")
        altitude_input = st.number_input("Enter maximum altitude (in meters):", min_value=1000, max_value=8000, key="cost_altitude")

    with col2:
        fitness_level_input = st.slider("Select your fitness level:", 1, 5, key="fitness_slider_2")
        group_size_input = st.number_input("Enter group size:", min_value=1, max_value=100, key="group_size_cost")

    if st.button('Predict Trek Cost', key="cost_button"):
        input_data_cost = np.array([[time_input, altitude_input, fitness_level_input, group_size_input]])
        input_scaled_cost = cost_scaler.transform(input_data_cost)  # Use the correct cost scaler
        predicted_cost = rf_cost_model.predict(input_scaled_cost)
        st.subheader(f"Predicted Trek Cost: ${predicted_cost[0]:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# DATA VISUALIZATIONS (Fourth Tab)
# ---------------------------------------------------

with tabs[3]:
    st.header("Trekking Data Visualizations")

    # Description for Trek Costs Over Time
    st.subheader("Trek Costs Over Time")
    st.markdown("""
    <div class="custom-intro">
    This line graph shows the average trekking costs over the years. It helps you understand 
    how trekking prices have evolved over time, possibly due to inflation, demand, or changes in trek services. 
    Hover over the graph to see exact values for each year.
    </div>
    """, unsafe_allow_html=True)
    cost_time_df = dataset.groupby('year')['Cost'].mean().reset_index()
    fig = px.line(
        cost_time_df, 
        x='year', 
        y='Cost', 
        title="Average Trek Cost Over the Years",
        labels={'year': 'Year', 'Cost': 'Average Trek Cost (USD)'},
        markers=True,
        line_shape="spline"  # Smooth transitions
    )
    fig.update_traces(line=dict(color="royalblue", width=3))  # Modern color and width
    fig.update_layout(
        template="plotly_dark",  # Modern dark theme
        hovermode="x unified",   # Unified hover effect
        title_font_size=20,      # Bigger title for emphasis
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )
    st.plotly_chart(fig)

    # Description for Trek Cost vs Group Size
    st.subheader("Trek Cost vs Group Size")
    st.markdown("""
    <div class="custom-intro">
    This scatter plot shows the relationship between trek costs and group size. You can explore how different group sizes 
    impact the overall trek cost. Hover over each point to see additional details such as the trek name, altitude, and weather conditions. 
    </div>
    """, unsafe_allow_html=True)
    fig = px.scatter(
        dataset, 
        x='Trekking_GroupSize', 
        y='Cost', 
        title="Trek Cost by Group Size",
        labels={'Trekking_GroupSize': 'Group Size', 'Cost': 'Trek Cost (USD)'},
        hover_data=['Trek', 'Max_Altitude', 'Weather_Conditions'],  # Additional hover data
        color='Rating',  # Add color based on Rating to make it more vibrant
        color_continuous_scale=px.colors.sequential.Viridis  # Attractive color scale
    )
    fig.update_layout(
        template="plotly_white",  # Clean modern white theme
        hovermode="closest",      # Detailed hover for each point
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )
    st.plotly_chart(fig)

    # Description for Age vs Trek Cost
    st.subheader("Interactive Age vs. Trek Cost")
    st.markdown("""
    <div class="custom-intro">
    This interactive scatter plot helps you understand how trek costs vary by age. The size of each point reflects the fitness level of the trekkers, 
    while the colors represent different genders. You can hover over each point to get more details about the trek, rating, and fitness level.
    </div>
    """, unsafe_allow_html=True)
    fig = px.scatter(
        dataset, 
        x='Age', 
        y='Cost', 
        title="Trek Cost by Age", 
        labels={'Age': 'Age', 'Cost': 'Trek Cost (USD)'},
        hover_data=['Trek', 'Rating', 'Fitness_Level'],
        size='Fitness_Level',  # Adding size for Fitness Level to differentiate points
        color='Sex',  # Different colors based on gender
        color_discrete_sequence=px.colors.qualitative.Pastel  # Soft pastel colors
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )
    st.plotly_chart(fig)

    # Description for Best Travel Time Frequency
    st.subheader("Best Travel Time Frequency")
    st.markdown("""
    <div class="custom-intro">
    This bar chart illustrates the frequency of the best travel times for trekking. It provides insights into the most popular months 
    for trekking activities based on the dataset. Hover over the bars to see the exact frequency for each time period.
    </div>
    """, unsafe_allow_html=True)
    best_travel_df = dataset['Best_travel_time_1'].value_counts().reset_index()
    best_travel_df.columns = ['Best Travel Time', 'Frequency']
    fig = px.bar(
        best_travel_df, 
        x='Best Travel Time', 
        y='Frequency', 
        title="Best Time to Travel by Frequency",
        labels={'Best Travel Time': 'Best Travel Time', 'Frequency': 'Frequency'},
        color='Frequency',  # Color bars based on frequency
        color_continuous_scale=px.colors.sequential.Plasma  # Bright color scale
    )
    fig.update_layout(
        template="plotly_dark", 
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )
    st.plotly_chart(fig)

    st.markdown('</div>', unsafe_allow_html=True)
