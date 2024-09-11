# AI-Enhanced-Tourism-Recommendation-System

Overview
The AI-Enhanced Tourism Recommendation System is a web-based application designed to provide personalized travel recommendations. By leveraging machine learning and natural language processing (NLP) techniques, the system suggests tailored destinations, activities, and accommodations based on user preferences such as budget, travel type, and desired experiences.

This project is developed using Python, Streamlit, and scikit-learn, offering a simple and intuitive interface for users to receive recommendations dynamically. The system is designed to help travelers make informed decisions by providing them with destinations that align with their personal interests and past behavior.

Features
Personalized Recommendations: The system generates custom travel recommendations based on user input, such as budget, travel type, and destination preferences.
AI-Powered: Utilizes TF-IDF vectorization and cosine similarity to recommend destinations based on user preferences and destination features.
Interactive Web App: Built with Streamlit, allowing users to input their preferences and receive real-time recommendations.
Scalable: The app is designed to scale easily and can be extended with more features like reviews, real-time updates, and trend analysis.
Tech Stack
Python 3.x
Streamlit (for the user interface)
scikit-learn (for machine learning)
Pandas (for data manipulation)
Pillow (for handling images)

streamlit run app.py
This will start the app, and you can open the link generated (usually http://localhost:8501/) in your browser.

How It Works
User Input: The user inputs their preferences (e.g., budget, type of travel, desired activities) via the web interface.
Model Processing: The app uses TF-IDF Vectorization and cosine similarity to analyze user preferences and match them to the most relevant destinations in the dataset.
Recommendations: The system returns a list of recommended destinations that best fit the user’s input, along with details such as budget category, type of destination, and user ratings.
