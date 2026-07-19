# backend/streamlit_app.py
import streamlit as st

st.title("Investor Risk Profiler")

# Inputs
goal = st.selectbox("Investment Goal", ["Wealth Building", "Retirement", "Education"])
horizon = st.slider("Investment Horizon (years)", 1, 30)
income = st.number_input("Annual Income (₹)")
risk = st.radio("Risk Appetite", ["Low", "Medium", "High"])

if st.button("Submit"):
    st.success(f"Risk profile calculated for goal: {goal}")
