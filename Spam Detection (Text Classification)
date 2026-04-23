from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

categories = ['sci.space', 'rec.sport.hockey']
data = fetch_20newsgroups(subset='all', categories=categories)

X = data.data
y = data.target

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
