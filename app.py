import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title("Classifying Iris Flower")
st.markdown("Toy model to classify Iris flowers into setosa, versicolor, and virginica.")
st.header("Plant Features")

col1,col2 = st.columns(2)

with col1:
    st.write("Sepal Characteristics")
    sepal_length = st.slider("Sepal Length", 0.0, 10.0, 0.5)
    sepal_width = st.slider("Sepal Width", 0.0, 10.0, 0.5)

with col2:
    st.write("Petal Characteristics")
    petal_length = st.slider("Petal Length", 0.0, 10.0, 3.0)
    petal_width = st.slider("Petal Width", 0.0, 10.0, 3.0)

iris_types = ["setosa", "versicolor", "virginica"]
if st.button("Predict Iris Type"):
    predicted_result = predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    index = predicted_result[0]
    st.text(iris_types[index])
    st.success("Predicted result is {}".format(iris_types[index]), icon="🎀")