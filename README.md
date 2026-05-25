
# Hotel Booking Cancellation Prediction System

## Problem Statement

The hospitality industry faces significant revenue loss due to hotel booking cancellations. Last-minute cancellations lead to empty rooms, inaccurate demand forecasting, inefficient resource allocation, and operational instability. Hotels often struggle to identify high-risk bookings early enough to take preventive actions such as dynamic pricing adjustments, overbooking strategies, personalized customer engagement, or deposit enforcement.

This project aims to solve this real-world business problem by building a machine learning-powered Hotel Booking Cancellation Prediction System capable of predicting whether a booking is likely to be canceled based on customer behavior, booking patterns, pricing, and operational factors.

---

# Real-World Impact

By accurately predicting booking cancellations, hotels can:

* Reduce revenue loss caused by vacant rooms
* Improve occupancy and resource planning
* Optimize pricing and booking strategies
* Identify high-risk customer behavior patterns
* Enhance customer retention and operational efficiency
* Support data-driven business decisions

This project demonstrates how data science and machine learning can directly improve decision-making in the hospitality industry.

---

# Project Workflow

## 1. Data Collection & Understanding

* Imported and explored the hotel booking dataset
* Understood business meaning of each feature
* Identified categorical, numerical, temporal, and operational variables

---

## 2. Exploratory Data Analysis (EDA)

### Univariate Analysis

Performed detailed analysis of individual features to understand:

* Customer demographics
* Booking behavior
* Stay duration patterns
* Seasonal trends
* Pricing distribution (ADR)
* Room allocation behavior
* Market segments and booking channels

### Bivariate Analysis

Analyzed relationships between features and booking cancellations:

* Lead time vs cancellation
* Deposit type impact
* Market segment behavior
* OTA vs direct bookings
* Loyalty and repeat guest behavior
* Pricing influence on cancellation probability

### Multivariate Analysis

Studied combined feature interactions to uncover:

* High-risk booking combinations
* Seasonal cancellation patterns
* Customer behavior clusters
* Operational risk indicators
* Revenue and pricing interactions

---

# Feature Engineering

Created advanced engineered features to improve predictive performance, including:

* Booking risk indicators
* Pricing-based features
* Customer behavioral patterns
* Lead-time categorization
* Previous cancellation ratios
* Room allocation change indicators
* Stay-based aggregated metrics

These engineered features allowed the model to capture complex booking behaviors more effectively.

---

# Machine Learning Workflow

## Data Preprocessing

* Missing value handling
* Encoding categorical variables
* Feature scaling
* Outlier handling
* Pipeline creation for reproducibility

## Model Development

Implemented and compared multiple machine learning algorithms:

* Logistic Regression
* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)
* K-Nearest Neighbors (KNN)

## Model Evaluation

Evaluated models using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

After extensive experimentation, Logistic Regression was selected as the final model due to its:

* Strong recall performance
* Stable generalization
* High ROC-AUC score
* Better balance between precision and recall

---

# Deployment

Built and deployed an interactive web application using <entity>Streamlit</entity>.

The application:

* Accepts real-time booking information
* Processes user inputs through the preprocessing pipeline
* Generates instant cancellation predictions
* Provides a clean and user-friendly interface

---

# Project Structure

```bash
hotel_booking_project/
│
├── data/
│   ├── raw_dataset.csv
│   └── feature_engineered_dataset.csv
│
├── notebooks/
│   ├── univariate_eda.ipynb
│   ├── bivariate_eda.ipynb
│   ├── multivariate_eda.ipynb
│   └── feature_engineering.ipynb
│
├── models/
│   ├── logistic_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Key Insights Discovered

* Long lead-time bookings have significantly higher cancellation probability
* Online Travel Agency (OTA) bookings are more cancellation-prone
* Non-refundable deposits reduce cancellations drastically
* Repeat guests are far less likely to cancel
* Dynamic pricing strongly influences booking behavior
* Room allocation changes can indicate operational risks
* Seasonal demand heavily impacts cancellation trends

---

# Achievements

* Completed end-to-end machine learning workflow
* Performed comprehensive EDA with business-focused insights
* Engineered high-impact predictive features
* Built and evaluated multiple classification models
* Selected and optimized the best-performing model
* Developed a fully functional deployment-ready web application
* Created a scalable preprocessing and prediction pipeline
* Demonstrated practical business application of machine learning in hospitality analytics

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* VS Code
* <entity>Streamlit</entity>

---

# Future Improvements

* Deploy application to cloud platforms
* Add real-time database integration
* Implement advanced ensemble models
* Introduce explainable AI dashboards
* Add booking recommendation systems
* Build automated hotel revenue optimization modules

---

# Conclusion

This project showcases the complete lifecycle of a real-world machine learning solution — from raw data exploration to deployment. By combining deep exploratory analysis, feature engineering, predictive modeling, and interactive deployment, the system provides a practical and scalable solution for predicting hotel booking cancellations and improving business decision-making in the hospitality sector.
# Author: Shorya Bisht
https://www.linkedin.com/in/shorya-bisht-a20144349/


