import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# APP CONFIGURATION & INFO DISPLAY
# --------------------------------

st.set_page_config(layout="wide")

st.title(":desktop_computer: Machine Learning Application")
st.markdown("""
This application allows you to apply various machine learning models to either an uploaded or selected sample dataset.
For all sample datasets, the application can advise you whether to use a regression or classification model, then provide
multiple modeling options for each. This application allows for you to experiment with hyperparameters and view changes
to model performance in real time, allowing for you to easily find the best models for whatever metrics you care about most.""")
st.write("  ") # empty space as separator
st.write("  ") # empty space as separator

col1, col2 = st.columns([1,3])

# -----------------
# DATASET SELECTION
# -----------------

# Step 1: Upload or Select a Dataset

with col1:
    st.subheader(":one: Upload or Select a Dataset")
    dataset = st.selectbox('Dataset selection', ['Diabetes', 'Breast Cancer', 'Iris', 'Wine', 'Upload Your Own'])

    if dataset == 'Diabetes': 
        appropriate_model_type = 'regression'
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Breast Cancer':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Iris':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Wine':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Upload Your Own':
            appropriate_model_type = 'none'
            uploaded_file = st.file_uploader("Upload a valid .csv file.", type=["csv"])
            df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a dataset or use the sample dataset to proceed.")
        st.stop()

# ---------------
# DATASET PREVIEW
# ---------------

# Show dataset
with col2:
    st.subheader(f"Dataset Preview: {dataset}")
    if dataset == 'Upload Your Own':
        st.info('Developer\'s Note: You must prepare the dataset for use in appropriate machine learning models prior to uploading. Otherwise, you may encounter errors when using the application\'s machine learning algorithms.')
    st.dataframe(df)

# ------------------
# DATA PREPROCESSING
# ------------------

with col1:
    st.subheader(":two: Select and Adjust Model")
    # Select target and features
    columns = df.columns.tolist()
    target_col = st.selectbox("Select the target column", columns)
    features = [col for col in columns if col != target_col]

X = df[features]
y = df[target_col]

# --------------------------
# MODEL TRAINING & SELECTION
# --------------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random state used to ensure consistent results

# Selecting a Model
with col1:
    model_name = st.selectbox("Select a model", ["Linear Regression", "Logistic Regression", "K-Nearest Neighbors", "Decision Tree"])
    if model_name == "Logistic Regression": # Classification
        model_type = "Classification"
        if appropriate_model_type == 'regression':
            st.warning("Please use a regression model with this dataset.")
        else:
            C = st.slider("C (Inverse of Regularization Strength)", 0.01, 10.0, 1.0)
            model = LogisticRegression(C=C, max_iter=1000)
    elif model_name == "K-Nearest Neighbors": # Classification
        model_type = "Classification"
        if appropriate_model_type == 'regression':
            st.warning("Please use a regression model with this dataset.")
        else:
            k = st.slider("Number of Neighbors (k)", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=k)
    elif model_name == "Decision Tree": # Classification & Regression
        model_type = "Classification"
        max_depth = st.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Minimum Samples Split", 2, 20, 2)
        min_samples_leaf = st.slider("Minimum Samples Leaf", 1, 20, 1)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    elif model_name == "Linear Regression": # Regression
        model_type = "Regression"
        if appropriate_model_type == 'classification':
            st.warning("Please use a classification model with this dataset.")
        else:
            model = LinearRegression()

# Encoding categorical target (if necessary for classification model)
if model_type == "Classification":
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif y.dtype in ['float64', 'float32'] and len(np.unique(y)) > 10:
        st.error("Your selected target appears to be continuous. Please select a categorical target column to use a classification model.")
        st.stop()
else:
    pass

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------
# VIEW RESULTS
# ------------
    
with col2:

    st.subheader("Model Visualization(s)")

    col2a, col2b = st.columns([1,1])

    with col2a:

        if model_type == "Classification":
            # Confusion matrix
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            # Plot actual vs predicted
            fig3, ax3 = plt.subplots()
            ax3.scatter(y_test, y_pred)
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax3.set_xlabel("Actual")
            ax3.set_ylabel("Predicted")
            ax3.set_title("Actual vs Predicted")
            st.pyplot(fig3)

    with col2b:
        if model_type == 'Classification':
            # ROC curve (if binary classification)
            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                st.write(f"**ROC AUC Score:** {auc:.2f}")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle='--')
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                ax2.legend()
                st.pyplot(fig2)
        else:
            pass

with col1:
    # Model Performance Metrics
    st.subheader("Model Performance Metrics")

    if model_type == "Classification":
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        precision = class_report['weighted avg']['precision']
        recall = class_report['weighted avg']['recall']
        f1_score = class_report['weighted avg']['f1-score']
        st.write("**Classification Report (Weighted):**")
        st.write(f'**Precision:** {precision:.2f}')
        st.write(f'**Recall (Sensitivity):** {recall:.2f}')
        st.write(f'**F1-Score:** {f1_score:.2f}')
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
    # Interpretation Guide
    if model_name == 'Linear Regression':
        st.info("Interpretation: In general, your model should aim for low MSE and RMSE scores, R² scores closer to 1.")
    else:
        st.info("Interpretation: In general, your model should aim for higher accuracy and F1-score (as a percent). You may prefer a higher precision when the cost of false positives is high, and a higher recall when the cost of a false negative is high. An ROC AUC Score further from 0.5 is best.")

