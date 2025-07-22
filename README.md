# 🧑‍💼 Employee Salary Predictor

Predict whether a person earns more than $50K per year using demographic features from the UCI Adult Census dataset. This ML web app is built using **XGBoost** and deployed with **Streamlit**.

---

## 🔍 Overview

This project applies machine learning to predict income class (`<=50K` or `>50K`) based on features like age, education, occupation, hours worked per week, etc.

The model is trained on the **UCI Adult Dataset** and deployed as an interactive web app using **Streamlit**.

---

## 📊 Features

- XGBoost-based classification model  
- Preprocessing pipeline with `ColumnTransformer`  
- Label encoding and feature scaling  
- Interactive web interface built with **Streamlit**  
- Real-time predictions  
- Deployed on Streamlit Cloud  

---

## 🧪 Tech Stack

- **Language:** Python 3  
- **Model:** XGBoost (`XGBClassifier`)  
- **Frontend:** Streamlit  
- **Libraries:** pandas, numpy, scikit-learn, joblib, matplotlib, seaborn  

---

## 📂 Repository Structure

```
employee-salary-predictor/
├── README.md # Project overview and documentation
├── app.py # Streamlit web app for prediction
├── emp_salary_prediction.ipynb # Jupyter notebook for model training
├── label_encoder.pkl # Saved label encoder
├── requirements.txt # List of dependencies
├── salary_prediction_pipeline.pkl # Trained XGBoost model pipeline
```

---

## 🚀 Getting Started (Local Setup)

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/employee-salary-predictor.git
   cd employee-salary-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Model Training (Optional)

To retrain the model:

```python
# Load and preprocess the UCI dataset
# Build preprocessing pipeline (ColumnTransformer)
# Fit XGBClassifier
# Save the model using joblib
joblib.dump(pipeline, "salary_prediction_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
```

---

## 📦 Deployment (Streamlit Cloud)

1. Push the repo to GitHub  
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Connect your GitHub  
4. Select the repo and set:
   - Main file: `app.py`
   - Python version: 3.11+
   - `requirements.txt` should include:
     ```
     pandas
     numpy
     scikit-learn
     xgboost
     streamlit
     joblib
     ```

---

## 📚 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Target Column:** `income` (`<=50K`, `>50K`)  
- **Features:** age, education, occupation, race, gender, etc.

---

## 🤝 Contributors

- **Anjithkrishnan K**

---

