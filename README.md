# Demand Prediction System

[![Published Research](https://img.shields.io/badge/Research-Published%20on%20ResearchGate-00CCBB)](https://doi.org/10.13140/RG.2.2.13036.07042)
[![University](https://img.shields.io/badge/University-of%20Nairobi-blue)](https://www.uonbi.ac.ke/)
[![Accuracy](https://img.shields.io/badge/R²%20Score-85%25-success)](https://doi.org/10.13140/RG.2.2.13036.07042)

> **🎓 Academic Research Project** | Supervised by Prof. Elisha Toyne O. Opiyo  
> **📄 Published Paper:** [Read on ResearchGate](https://doi.org/10.13140/RG.2.2.13036.07042)  
> **🏛️ Deployed At:** Huduma Center Makadara

This project predicts the demand for e-government services at **Makadara Huduma Center ** using machine learning.  

## 📊 Problem Statement

Huduma Centers across Kenya face challenges in resource allocation due to unpredictable service demand. 
Long waiting times and understaffing during peak periods lead to poor service delivery.

This project uses historical booking data (2020-2023) from  Makadara Huduma Center to:
- Predict quarterly demand for over 10 services categories
- Enable data-driven resource planning
- Reduce citizen waiting times

**Impact:** Helps government optimize staff scheduling and improve e-citizen service delivery.

## 🚀 Features
- Data preprocessing and cleaning  
- Exploratory Data Analysis   
- Machine Learning model for demand prediction  
- Django web application for deployment


## 📂 Project Structure
predict-demand/
├── 📁 data/                    # Datasets (training & testing)
├── 📁 demand/                  # Django app folder
│   ├── views.py               # Application logic
│   ├── models.py              # Database models
│   └── urls.py                # URL routing
├── 📁 prediction/              # ML model code
│   ├── settings.py            # Django settings
│   └── wsgi.py                # WSGI config
├── 📁 templates/               # HTML templates
│   ├── index.html             # Home page
│   ├── predict.html           # Prediction form
│   └── results.html           # Results display
├── 📁 static/                  # CSS, JS, images
├── 📁 models/                  # Saved ML models
│   ├── rf_regressor_model.pkl
│   └── encoder.joblib
├── 📄 db.sqlite3              # SQLite database
├── 📄 manage.py               # Django management
├── 📄 model_training.py       # Model training script
├── 📄 requirements.txt        # Dependencies
└── 📄 README.md               # This file

---
## 🛠️ Tech Stack

**Machine Learning:**
- Python 3.8+
- Scikit-learn (Random Forest Regressor)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)

**Web Application:**
- Django 4.x
- SQLite (development database)
- HTML/CSS/Bootstrap (frontend)

**Why Random Forest?**
- Handles non-linear relationships in booking patterns
- Robust to outliers and missing data
- Provides feature importance rankings
- Outperformed Linear Regression (R²: 0.85 vs 0.67) and Decision Trees (R²: 0.79)

## 🔄 How It Works

1. **Data Collection:** Historical booking data from Makadara Huduma Center (2020-2023)
2. **Preprocessing:** 
   - Cleaned missing values (dropped 3% incomplete records)
   - Encoded categorical variables (service types, quarters)
   - Feature engineering (lag features, rolling averages)
3. **Model Training:** 
   - Train/test split: 80/20
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation (5-fold)
4. **Deployment:** Django web app where users input service parameters and get demand predictions
5. **Output:** Predicted demand + confidence intervals + historical comparison

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/angelamawia/predict-demand.git
cd predict-demand
```
### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Run Database Migrations
```bash
python manage.py migrate
```
### Step 5: Train the Model (Optional - pre-trained model included)
```bash
python model_training.py
```
### Step 6: Start Development Server
```bash
python manage.py runserver
```
### Step 7: Access Application
Open your browser and navigate to: `http://127.0.0.1:8000/`  

## 🚀 Future Enhancements

- [ ] Deploy to AWS/Railway for public access
- [ ] Add role-based access
- [ ] Implement real-time predictions using streaming data
- [ ] Expand to multiple Huduma Centers (nationwide model)
- [ ] Add SMS/email alerts for predicted high-demand periods
- [ ] Integrate with Huduma Center booking system API
- [ ] Build mobile app version using React Native

## 📊 Dataset Information

**Source:** Huduma Center Makadara Booking Records  
**Period:** January 2020 - December 2023 (4 years)  
**Records:** 12,847 bookings  
**Features:**
- Service type (8 categories: ID, Birth Certificate, Passport, etc.)
- Quarter (Q1-Q4)
- Year (2020-2023)
- Historical demand patterns
- Derived features (moving averages, lag variables)

**Data Privacy:** All personal information removed. Only aggregate booking counts used.

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas needing help:**
- Frontend UI/UX improvements
- Additional visualization types
- Testing coverage
- Documentation

## 👩‍💻 Author

**Angela Mawia Charles**  
Data Scientist | Python, ML, Django

- 📧 Email: angelmawia.01@gmail.com
- 💼 LinkedIn: [angela-mawia](https://www.linkedin.com/in/angela-mawia-a2114b213/)
- 🐙 GitHub: [@angelamawia](https://github.com/angelamawia)
- 📝 Portfolio: [Notion Portfolio](https://www.notion.so/Angela-Mawia-Data-Scientist-256b522b2fc680ef9349d2f78de32efb)

---

## 🙏 Acknowledgments

- Huduma Kenya for data access
- University of Nairobi Computer Science Department
- Open-source contributors of Scikit-learn and Django

---

## 📞 Contact & Support

Questions or suggestions? Reach out:
- Open an [issue](https://github.com/angelamawia/predict-demand/issues)
- Email: angelmawia.01@gmail.com
- LinkedIn message

**⭐ If you find this project useful, please star the repository!**

---

*Last updated: November 2025*
