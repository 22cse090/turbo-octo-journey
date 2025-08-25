import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

st.title("🤖 AutoML Buddy - TEACHABLE MACHINE 🤯🧩")

# Upload CSV
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.write("Preview:", df.head())

    target = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    task_type = "classification" if y.dtype == "object" else "regression"
    st.write(f"Detected task: {task_type.title()}")

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if task_type == "classification":
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task_type == "classification":
            st.write("Accuracy:", accuracy_score(y_test, preds))
        else:
            st.write("MSE:", mean_squared_error(y_test, preds))

        joblib.dump(model, "trained_model.pkl")
        st.success("Model trained and saved!")
