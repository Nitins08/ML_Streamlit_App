import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model and dataset
model = joblib.load("iris_model.pkl")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

st.title("Iris Flower Prediction & Data Analysis")

# Input features
st.subheader("Input Measurements")
col1, col2 = st.columns(2)
with col1:
    sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1)
    sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1)
with col2:
    pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1)
    pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)

# Prediction
if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: {flowers[pred[0]]}")

# Scatter Plot Visualization
st.subheader("Feature Distribution Plot")
fig, ax = plt.subplots(figsize=(7, 4))

colors = {'setosa': 'tab:blue', 'versicolor': 'tab:orange', 'virginica': 'tab:green'}
for species_name, group in df.groupby('species'):
    ax.scatter(group['sepal length (cm)'], group['petal length (cm)'], label=species_name, color=colors[species_name], alpha=0.8)

# Highlight current input point
ax.scatter(sl, pl, color='red', marker='X', s=120, label='Current Input')

ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

st.pyplot(fig)