import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle, os

print("✅ train_model.py started")

# ▶️ 1) Build a tiny sample dataset in-memory
data = [
    ("The economy is stable and growing", "REAL"),
    ("Celebrity found alive on Mars!",     "FAKE"),
    ("New study shows benefits of tea",    "REAL"),
    ("Scientists confirm unicorn DNA",      "FAKE"),
    ("Elections proceed peacefully",        "REAL"),
    ("Time traveler speaks to press",       "FAKE")
]
df = pd.DataFrame(data, columns=["text","label"])
print(f"📊 Sample dataset created: {len(df)} rows")

# ▶️ 2) Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.5, random_state=42
)
print(f"✂️  Split: {len(x_train)} train / {len(x_test)} test")

# ▶️ 3) Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec  = vectorizer.transform(x_test)
print("🔢 Vectorization done")

# ▶️ 4) Train
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_vec, y_train)
print("🏋️  Model trained on sample data")

# ▶️ 5) Save artifacts
os.makedirs("backend", exist_ok=True)
pickle.dump(model, open("backend/model.pkl","wb"))
pickle.dump(vectorizer, open("backend/vectorizer.pkl","wb"))
print("💾 Saved model.pkl + vectorizer.pkl in backend/")

# ▶️ 6) Evaluate
y_pred   = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Sample accuracy: {accuracy:.2f}")
