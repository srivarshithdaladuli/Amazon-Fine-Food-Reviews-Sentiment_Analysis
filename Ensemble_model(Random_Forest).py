import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import string, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub("<.*?>", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@st.cache_data
def load_and_clean_data():
    data = pd.read_csv("Reviews.csv", encoding="utf-8-sig")
    data = data.sample(n=75000, random_state=42)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    data.dropna(subset=['Text', 'Score'], inplace=True)
    data['cleaned_review'] = data['Text'].apply(clean_text)
    data['review_length'] = data['cleaned_review'].apply(lambda x: len(x.split()))
    
    # Create new sentiment label
    def get_sentiment(score):
        if score in [4, 5]:
            return "Positive"
        elif score == 3:
            return "Neutral"
        else:
            return "Negative"
    
    data['SentimentLabel'] = data['Score'].apply(get_sentiment)
    data.drop(columns=['Score'], inplace=True)  # Drop original score
    return data

data = load_and_clean_data()

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review'])
y = data['SentimentLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {k: v for k, v in zip(np.unique(y_train), class_weights)}

rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("🍽 Amazon Food Reviews Sentiment Analysis (Multiclass Classification)")

st.write("### 🧮 Distribution of Sentiment Labels")
st.bar_chart(data['SentimentLabel'].value_counts())

st.write("### 📝 Review Length Distribution")
st.bar_chart(data['review_length'])

st.write("### ☁ WordClouds by Sentiment (Score)")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, label in enumerate(["Negative", "Neutral", "Positive"]):
    text = " ".join(data[data['SentimentLabel'] == label]['cleaned_review'].tolist())
    wordcloud = WordCloud(width=500, height=600, background_color='white').generate(text)
    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(f"Sentiment: {label}")
st.pyplot(fig)

st.write("### Accuracy")
st.text(accuracy_score(y_test, y_pred))

st.write("### Precision")
st.text(f"Macro Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")


st.write("### 🔁 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, labels=rf.classes_)

st.write("### ⭐ Top Feature Importances")
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]
features = np.array(vectorizer.get_feature_names_out())[indices]
st.bar_chart(pd.Series(importances[indices], index=features))

data['ProductId'] = data['ProductId'].astype('category')

st.write("### 🏷️ Top Reviewed Products")

top_products = data['ProductId'].value_counts().head(20)
st.bar_chart(top_products)

# Optional table view
if st.checkbox("Show Top Product IDs with Review Counts"):
    st.dataframe(top_products.reset_index().rename(columns={'index': 'ProductId', 'ProductId': 'ReviewCount'}))



