# Demand Prediction System

This project predicts the demand for e-government services at **Huduma Center Makadara** using machine learning.  
The goal is to help improve service delivery by forecasting demand and allocating resources more effectively.  

---

## 🚀 Features
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Machine Learning model for demand prediction  
- Django web application for deployment


## 📂 Project Structure
demand-prediction/
📁 .idea/                  → IDE configuration files  
📁 data/                   → Dataset(s) used for training/testing  
📁 demand/                 → Django app folder  
📁 prediction/              → Prediction-related code/app logic  
📁 templates/               → HTML templates for the web app  

📄 db.sqlite3               → SQLite database  
📄 encoder.joblib           → Saved encoder for categorical variables  
📄 label_encoders.py        → Script for encoding labels  
📄 manage.py                → Django project management file  
📄 model_training.py        → Script for training ML models  
📄 rf_regressor_model.py    → Random Forest regressor model  


---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/demand-prediction.git
   cd demand-prediction
🧠 Model

Built using Scikit-learn
Evaluated using metrics such as RMSE and R² Score
Best-performing model saved in models/ folder

🌐 Web App

The project includes a Django web app where users can input service details and view demand predictions.

📈 Results

The model achieved good accuracy in predicting demand patterns.

Helps allocate resources better and reduce waiting times.

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

👩‍💻 Author

Angela Mawia
📧 angelmawia.01@gmail.com
