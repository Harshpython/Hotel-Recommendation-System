import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import string

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('hotel_reviews.csv')

# Initialize stemmer
ps = PorterStemmer()

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Split into words
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]  # Remove stopwords and stem
    return ' '.join(words)

# Apply preprocessing
df['processed_review'] = df['review'].apply(preprocess_text)

# Create TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['processed_review'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping of hotel IDs to indices
hotel_indices = pd.Series(df.index, index=df['hotel_id']).drop_duplicates()

# Function to get recommendations
def get_recommendations(hotel_id, cosine_sim=cosine_sim):
    # Get the index of the hotel that matches the hotel_id
    idx = hotel_indices[hotel_id]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar hotels
    sim_scores = sim_scores[1:11]

    # Get the hotel indices
    hotel_indices_recommended = [i[0] for i in sim_scores]

    # Return the top 10 most similar hotels
    return df['hotel_id'].iloc[hotel_indices_recommended]

# Example usage
print(get_recommendations(hotel_id=1))  # Replace 1 with a valid hotel_id from your dataset
