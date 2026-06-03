
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_and_save_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv('train.csv').fillna('')
    except FileNotFoundError:
        print("Error: 'train.csv' not found.")
        return

    # The ISOT combined dataset already has a 'content' column (title + text)
    # We will use that directly.
    
    print("Cleaning text...")
    def clean_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        return text
      
    df['content'] = df['content'].apply(clean_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(df['content'])
    Y = df['label']
    
    print("Training model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, Y)
    
    print("Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Done! Model saved to 'model.pkl' and 'vectorizer.pkl'")

if __name__ == "__main__":
    train_and_save_model()
