import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score

# Title
st.title("Machine Learning Algorithms Visualization")

# Sidebar for user input
st.sidebar.header("Upload Your Dataset or Use Default")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

algorithm_name = st.sidebar.selectbox(
    "Select Algorithm",
    ("Logistic Regression", "KNN", "SVM", "Decision Tree", "K-Means", "DBSCAN")
)

# Load dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset")
    st.write(data.head())

    # User selects the target column
    target_column = st.sidebar.selectbox("Select Target Column", data.columns)

    # Features and Target
    X = data.drop(columns=[target_column])
    y = data[target_column]
else:
    from sklearn import datasets
    data = datasets.load_iris()
    X, y = data.data, data.target
    X = pd.DataFrame(X, columns=data.feature_names)

# Handle Categorical Columns
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    st.write("Encoding categorical columns:", categorical_cols.tolist())
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])  # Convert text to numbers

# Handle Missing Values
X.fillna(X.mean(), inplace=True)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split (for supervised algorithms)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression()
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "SVM":
        return SVC(kernel='linear')
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "K-Means":
        return KMeans(n_clusters=len(set(y)), random_state=42)
    elif name == "DBSCAN":
        return DBSCAN(eps=0.5, min_samples=5)

model = get_model(algorithm_name)

# Training and prediction (for supervised learning)
if algorithm_name in ["Logistic Regression", "KNN", "SVM", "Decision Tree"]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {acc:.2f}")

    # Create a DataFrame with actual vs predicted values
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    
    # Display results
    st.write("### Predictions")
    st.write(results_df.head())

    # Add a download button for predictions
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv",
    )

# Visualization
fig, ax = plt.subplots()
if algorithm_name in ["K-Means", "DBSCAN"]:
    y_pred = model.fit_predict(X_scaled)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette="viridis", ax=ax)
    ax.set_title(f"{algorithm_name} Clustering")
else:
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm", ax=ax)
    ax.set_title(f"True Labels Visualization")

st.pyplot(fig)

# Add a download button for processed dataset
processed_data = pd.DataFrame(X_scaled, columns=X.columns)
csv_processed = processed_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Processed Dataset",
    data=csv_processed,
    file_name="processed_dataset.csv",
    mime="text/csv",
)
