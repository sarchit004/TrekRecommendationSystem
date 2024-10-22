import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mockup functions for testing
def get_recommendations(input_data):
    start_time = time.time()
    # Simulate some processing delay
    time.sleep(np.random.uniform(0.1, 1.0))  
    latency = time.time() - start_time
    # Mockup recommendation results
    recommendations = {"Recommendation": "Trek X", "Cost": "$2000"}
    return recommendations, latency

def evaluate_model(predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    rmse = mean_squared_error(true_values, predictions, squared=False)
    return mae, rmse

# System Testing Dashboard
st.title("System Testing Dashboard")

# Latency Testing
st.subheader("Latency Testing")
input_data = np.random.rand(1, 5)  # Mockup input data
recommendations, latency = get_recommendations(input_data)
st.write(f"Latency: {latency:.2f} seconds")
st.write("Recommendations:", recommendations)

# Track latency over multiple requests
latencies = [get_recommendations(input_data)[1] for _ in range(10)]
st.line_chart(latencies)

# Accuracy Testing
st.subheader("Model Accuracy Testing")
# Mockup predictions and true values
predictions = np.random.rand(10)
true_values = np.random.rand(10)
mae, rmse = evaluate_model(predictions, true_values)

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualize accuracy metrics
st.bar_chart({"MAE": [mae], "RMSE": [rmse]})

# Reliability Testing (Simulate load)
st.subheader("Reliability Testing")
simulated_users = st.slider("Simulate number of users:", 1, 50, 10)
response_times = [get_recommendations(input_data)[1] for _ in range(simulated_users)]

# Visualize reliability under load
fig = px.line(x=range(simulated_users), y=response_times, labels={"x": "User", "y": "Response Time (s)"}, title="System Response Times Under Load")
st.plotly_chart(fig)
