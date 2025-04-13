import pandas as pd
import random

# Sample feedback templates
positive_phrases = [
    "I love the", "Amazing", "Very satisfied with the", "The best", "Happy with the", "Superb", "Great"
]
negative_phrases = [
    "I hate the", "Terrible", "Very disappointed with the", "Worst", "Unhappy with the", "Poor", "Bad"
]
topics = [
    "camera quality", "battery life", "customer service", "app performance", "UI/UX", "delivery speed",
    "packaging", "price", "audio quality", "features", "connectivity", "charging time", "build quality"
]

# Create synthetic feedbacks
def generate_feedback():
    sentiment = random.choices(["positive", "negative"], weights=[0.7, 0.3])[0]
    topic = random.choice(topics)
    if sentiment == "positive":
        phrase = random.choice(positive_phrases)
    else:
        phrase = random.choice(negative_phrases)
    return f"{phrase} {topic}."

# Generate 100,000 rows
data = {"feedback": [generate_feedback() for _ in range(100000)]}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sample_feedback.csv", index=False)
print("✅ sample_feedback.csv with 100,000 entries created.")
