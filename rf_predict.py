import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('RF.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "Age", "BMI", "AST", "GFR", "HDL-C", "FBG", "MAP"
]

# Streamlit user interface
st.title("Risk of Aortic Stiffening")

# Age: numerical input
Age = st.number_input("Age:", min_value=0, max_value=120, value=41)

BMI = st.number_input("BMI:", min_value=12.00, max_value=38.00, value=22.00, step=0.10)

AST = st.number_input("AST:", min_value=0, max_value=50, value=18)

GFR = st.number_input("GFR:", min_value=30, max_value=150, value=100)

HDL_C = st.number_input("HDL-C:", min_value=0.01, max_value=10.00, value=1.21, step=0.01)

FBG = st.number_input("FBG:", min_value=1.0, max_value=40.0, value=5.4, step=0.10)

MAP = st.number_input("MAP:", min_value=30, max_value=140, value=88)

# Process inputs and make predictions
feature_values = [Age, BMI, AST, GFR, HDL_C, FBG, MAP]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Aortic Atherosclerosis. "
            f"The model predicts that your probability of having Aortic Atherosclerosis is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Aortic Atherosclerosis. "
            f"The model predicts that your probability of not having Aortic Atherosclerosis is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
        mode='classification'
    )

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)