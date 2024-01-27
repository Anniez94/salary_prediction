import streamlit as st
import pickle
import numpy as np 
import pandas as pd


def load_model():
    with open("v1.pk1", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

reg = data["model"]
le_ed = data["le_ed"]
le_gender = data["le_gender"]
categories = data["df"]
min = data["min"]

df = pd.read_csv("Salary Data.csv")
df = df.dropna()

education = df["Education Level"].unique().tolist()
job_title = df["Job Title"].unique().tolist()
gender = df["Gender"].unique().tolist()


def show_predict_page():
    st.title("Salary Prediction Using Kaggle Dataset")

    st.write("### This is a beginner's guide using supervised learning to predict the salary")

    st.dataframe(df, use_container_width=True)

    st.write("This dataset has the following columns which are : Age, Gender, Education Level, Job Title, Years of experience and Salary. From this dataset we can deduce that the 'salary' is what we are going to be predicting.")

    ed = st.selectbox("##### Education Level", education)

    jt = st.selectbox("##### Job Title", job_title)

    gn = st.selectbox("##### Gender", gender)

    age = st.slider("Age", 18, 60)

    experience = st.number_input("Years of Experience", 0, 42)

    ok = st.button("Predict Salary", use_container_width=True)

    if ok:
        result = predict_salary(age, gn, ed, experience, jt)
        st.subheader(f"The estimated salary is {result[0]:.2f}")


def predict_salary(age, gn, ed, experience, jt):
    category = categories[categories["Job Title"] == jt]["Category"].values[0]

    # X = np.array([[age, gn, ed, experience, category]])
    X = pd.DataFrame.from_dict({"Age":[age], "Gender": [gn], "Education Level": [ed], "Years of Experience": [experience], "Category": [category]})
    X["Gender"] = le_gender.transform(X["Gender"])
    X["Education Level"] = le_ed.transform(X["Education Level"])
    X = X.astype(float)
    X = min.transform(X)

    salary = reg.predict(X)
    return salary
