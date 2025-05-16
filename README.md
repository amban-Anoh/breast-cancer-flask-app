# Breast Cancer Prediction App

This project demonstrates a full machine learning deployment pipeline using Flask, scikit-learn, and Heroku.

---

## Project Summary

- **Data Preparation & Training:**  
  A classification model was trained using `scikit-learn` on breast cancer feature data.

- **Model Serialization:**  
  The trained model was saved as a `.pkl` file using Python's `pickle` module.

- **Flask Web App:**  
  A web interface built with Flask allows users to input 9 features and receive a prediction (Benign or Malignant).

- **Production Deployment:**  
  The app is deployed to **Heroku** using **Gunicorn** and a full **CI/CD pipeline** via **GitHub Actions**.

- **Health Check:**  
  An endpoint at `/ping` confirms that the app is running and responsive.

---

## 📎 Useful Links

- **GitHub Repository:** [https://github.com/amban-Anoh/breast-cancer-flask-app](https://github.com/amban-Anoh/breast-cancer-flask-app)
- **Live Heroku App:** [https://aborobot-tech-2025-551cac8bd375.herokuapp.com](https://aborobot-tech-2025-551cac8bd375.herokuapp.com)
