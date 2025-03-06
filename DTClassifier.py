import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# Generate Sample Data
X, y = X, y = make_blobs(
    n_samples=1000, 
    centers=[(-0.5, 0.5), (0.5, -0.5)],  # Center the blobs in opposite quadrants
    cluster_std=0.4,  # Controls spread of the clusters
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("Decision Tree Classifier")

# Sidebar for Hyperparameters
st.sidebar.header("Hyperparameters")
criterion=st.sidebar.selectbox("criterion",["gini", "entropy", "log_loss"])
splitter=st.sidebar.selectbox("splitter",["best", "random"])
max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=10, value=2, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", 1, len(y_train), 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, len(y_train), 1)
max_features = st.sidebar.slider("Max Features", 1, X_train.shape[1], 2)
max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", min_value=2, max_value=50, value=10, step=1)
min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Run Algorithm Button
if st.sidebar.button("Run Algorithm"):
    
    # Train Decision Tree Model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes is not None else None,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Display Accuracy
    st.write(f"Accuracy for Decision Tree: {clf.score(X_test, y_test):.2f}")

    # Plot Decision Boundaries
    def plot_decision_boundary(model, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", edgecolor="k")
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        st.pyplot(plt)

    plot_decision_boundary(clf, X_train, y_train)


    # Visualize Decision Tree
    dot_data = export_graphviz(clf, feature_names=["Col1", "Col2"], class_names=["Class 0", "Class 1"], filled=True, rounded=True)
    st.graphviz_chart(dot_data)

    
