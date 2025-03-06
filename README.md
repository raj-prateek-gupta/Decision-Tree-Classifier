# Decision Tree Classifier - Streamlit App

## 📌 Overview
This project is a **Streamlit-based Decision Tree Classifier** that allows users to:
- Tune hyperparameters such as `max_depth`, `criterion`, `splitter`, etc.
- Visualize the decision tree structure and decision boundaries
- Display the model’s accuracy score

## 🚀 Features
✔ **Hyperparameter Tuning** - Adjust decision tree parameters dynamically
✔ **Decision Tree Visualization** - Graph representation of the trained tree
✔ **Accuracy Score Display** - Evaluate model performance
✔ **Decision Boundary Visualization** - See how the model classifies data

## 🛠 Installation & Setup
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/raj-prateek-gupta/Decision-Tree-Classifier
cd decision-tree-classifier
```
### 2️⃣ Install required dependencies:
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```
✅ If streamlit is not recognized:
```bash
python -m streamlit run app.py
```


# Introduction to Decision Tree Classifier

## 📖 What is a Decision Tree Classifier?
A **Decision Tree Classifier** is a **supervised machine learning algorithm** used for **classification tasks**. It works by splitting the dataset into smaller subsets based on feature values, forming a tree-like structure. Each internal node represents a decision based on a feature, and each leaf node represents a class label.

## 🔹 How It Works
1. The dataset is recursively split based on feature values.
2. The best split is determined using measures like **Gini, Impurity** or **Entropy**.
3. The process continues until a stopping criterion is met (e.g., max depth reached).
4. The final tree structure is used to make predictions.

## ⚙ Key Hyperparameters
- **`criterion`**: The function used to measure the quality of a split (`gini`, `entropy`, `log_loss`)
- **`splitter`**: How the split is chosen (`best`, `random`)
- **`max_depth`**: The maximum depth of the tree
- **`min_samples_split`**: The minimum number of samples required to split a node
- **`min_samples_leaf`**: The minimum number of samples at a leaf node

## ✅ Advantages
✔ Easy to understand and interpret
✔ Handles both numerical and categorical data
✔ Requires little data preprocessing

## ❌ Disadvantages
✖ Prone to overfitting
✖ Can create biased trees if dataset is imbalanced









