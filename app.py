import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Tour Recommendor", page_icon="🛫")


# synthetic data
data = {
    'Destination': [
        'Paris', 'Bali', 'Tokyo', 'Colorado', 'Kyoto', 'Goa', 'Sydney',
        'Rome', 'New York', 'Santorini', 'Cape Town', 'Barcelona', 'Singapore',
        'Dubai', 'Amsterdam', 'Maui', 'Istanbul', 'Reykjavik', 'Lisbon', 'Hanoi'
    ],
    'Type': [
        'Cultural', 'Beach', 'Cultural', 'Adventure', 'Cultural', 'Beach', 'Nature',
        'Cultural', 'Urban', 'Beach', 'Nature', 'Cultural', 'Urban',
        'Luxury', 'Cultural', 'Beach', 'Cultural', 'Nature', 'Urban', 'Cultural'
    ],
    'Geography': [
        'International', 'International', 'International', 'Domestic', 'International', 'Domestic', 'International',
        'International', 'International', 'International', 'International', 'International', 'International',
        'International', 'International', 'International', 'International', 'International', 'International', 'International'
    ],
    'Age Group': [
        '26-35', '18-25', '36-50', '18-25', '26-35', '18-25', '26-35',
        '26-35', '26-35', '18-25', '26-35', '26-35', '26-35',
        '26-35', '26-35', '18-25', '26-35', '36-50', '26-35', '18-25'
    ],
    'Traveler Type': [
        'Family', 'Solo', 'Family', 'Solo', 'Solo', 'Solo', 'Family',
        'Solo', 'Family', 'Solo', 'Family', 'Solo', 'Family',
        'Solo', 'Solo', 'Solo', 'Family', 'Solo', 'Solo', 'Solo'
    ],
    'Budget': [
        'High', 'Medium', 'High', 'Low', 'Medium', 'Low', 'High',
        'High', 'High', 'Medium', 'High', 'Medium', 'High',
        'High', 'Medium', 'Medium', 'High', 'High', 'Medium', 'Low'
    ],
    'Rating': [
        4.5, 4.7, 4.8, 4.6, 4.9, 4.3, 4.7,
        4.6, 4.8, 4.5, 4.7, 4.6, 4.8,
        4.9, 4.7, 4.6, 4.8, 4.5, 4.7, 4.4
    ],
    'Preferences': [
        'Luxury, Culture', 'Adventure, Beach', 'Culture, Luxury', 'Adventure, Budget',
        'Culture, Relaxation', 'Budget, Beach', 'Nature, Relaxation',
        'Culture, History', 'Urban, Nightlife', 'Beach, Relaxation',
        'Nature, Adventure', 'Culture, Food', 'Urban, Shopping',
        'Luxury, Shopping', 'Culture, Art', 'Beach, Adventure',
        'Culture, Heritage', 'Nature, Exploration', 'Urban, Culture', 'Culture, Exploration'
    ]
}

df = pd.DataFrame(data)

# Here we are Combining relevant features into a single string
df['Combined_Features'] = df.apply(lambda row: f"{row['Type']} {row['Geography']} {row['Age Group']} {row['Traveler Type']} {row['Budget']} {row['Preferences']}", axis=1)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['Destination']).drop_duplicates()

def recommend(destination, user_preferences, top_n=5):
    if destination not in indices:
        return "Destination not found in the dataset."

    idx = indices[destination]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    destination_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[destination_indices][['Destination', 'Type', 'Budget', 'Rating']]

    if user_preferences:
        user_pref_vector = tfidf.transform([user_preferences])
        user_sim = cosine_similarity(user_pref_vector, tfidf_matrix).flatten()
        recommendations = recommendations.copy()
        recommendations['User_Similarity'] = user_sim[destination_indices]
        recommendations = recommendations.sort_values(by='User_Similarity', ascending=False)

    return recommendations[['Destination', 'Type', 'Budget', 'Rating']]

# Our Streamlit App
st.title("AI-Enhanced Tourism Recommendation System")

# User inputs
destination = st.selectbox('Select a Destination', df['Destination'].values)
user_preferences = st.text_input('Enter your preferences (e.g., Luxury, Culture)')

if st.button('Get Recommendations'):
    recommendations = recommend(destination, user_preferences)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.write("### Recommended Destinations:")
        st.dataframe(recommendations)
