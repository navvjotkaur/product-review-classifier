import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load the dataset
df = pd.read_csv("reviews.csv")

# Step 2: Drop rows with missing reviews or ratings
df = df[['reviews.text', 'reviews.rating']].dropna()

# Step 3: Create sentiment labels
def get_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

df['label'] = df['reviews.rating'].apply(get_sentiment)

# Step 4: Filter out neutral reviews (optional)
df = df[df['label'] != 'neutral']

# Step 5: Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned'] = df['reviews.text'].apply(clean_text)

# Step 6: Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Step 7: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")
# Step 9: Evaluate model
accuracy = model.score(X_test, y_test)
print(f"📊 Model accuracy: {accuracy:.2f}")
# Step 10: Print dataset information
print("📊 Dataset information:")
print(f"Total reviews: {len(df)}")
print(f"Positive reviews: {len(df[df['label'] == 'positive'])}")
print(f"Negative reviews: {len(df[df['label'] == 'negative'])}")
# Step 11: Show first few rows of the cleaned dataset
print("\n📊 First 5 rows of cleaned dataset:")
print(df[['cleaned', 'label']].head())