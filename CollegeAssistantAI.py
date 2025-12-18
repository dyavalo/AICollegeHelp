# This is an improved AI assistant in Python with real Machine Learning added to make it more like actual AI, not just a chatbot.
# I added scikit-learn for a simple regression model to predict admission chances based on GPA and portfolio features.
# Trained on a small made-up dataset, but in real life could load from CSV or something.
# I know the code is simple and not cool, but I think the idea is more important here.
# For ML, it extracts basic features from portfolio like word count and keywords.
# Dependencies: pip install scikit-learn difflib  # Assuming sklearn is installed.
# Run this in console, interacts via input().
# Some parts are redundant, like saving data multiple times, but it works.

import difflib
import random
import pickle
import os
from sklearn.linear_model import LinearRegression  # For the ML part.
from sklearn.model_selection import train_test_split  # To split data.
import numpy as np  # For arrays.

# File for saving user data.
DATA_FILE = 'ai_user_data.pkl'

# Hard-coded university database.
universities = {
    'Harvard University': {'fields': ['Computer Science', 'Business Administration', 'Medicine'], 'level': 'Bachelor', 'avg_gpa': 3.9, 'tuition': 50000, 'acceptance_rate': 5},
    'MIT': {'fields': ['Engineering', 'Computer Science', 'Physics'], 'level': 'Bachelor', 'avg_gpa': 3.95, 'tuition': 55000, 'acceptance_rate': 7},
    'Stanford University': {'fields': ['Business', 'Engineering', 'Computer Science'], 'level': 'Bachelor', 'avg_gpa': 3.9, 'tuition': 52000, 'acceptance_rate': 4},
    'University of California, Berkeley': {'fields': ['Computer Science', 'Business', 'Biology'], 'level': 'Bachelor', 'avg_gpa': 3.8, 'tuition': 14000, 'acceptance_rate': 17},
    'Oxford University': {'fields': ['Medicine', 'Law', 'Business'], 'level': 'Bachelor', 'avg_gpa': 3.7, 'tuition': 40000, 'acceptance_rate': 17},
    'Yale University': {'fields': ['Law', 'Medicine', 'Arts'], 'level': 'Bachelor', 'avg_gpa': 3.85, 'tuition': 48000, 'acceptance_rate': 6},
    'Boston University': {'fields': ['Business', 'Computer Science', 'Medicine'], 'level': 'Bachelor', 'avg_gpa': 3.7, 'tuition': 58000, 'acceptance_rate': 18},  # Added BU based on past talks.
}

# Load user data if exists.
user_data = {}
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        user_data = pickle.load(f)

# Simple ML model training - done once at start.
# Made-up data: GPA, portfolio_length, has_award (0/1), target_chance.
X = np.array([
    [3.5, 50, 0], [3.8, 100, 1], [4.0, 80, 1], [3.2, 30, 0], [3.9, 120, 1],
    [3.6, 60, 0], [3.7, 90, 1], [3.4, 40, 0], [3.95, 110, 1], [3.1, 20, 0]
])
y = np.array([40, 70, 90, 20, 85, 50, 65, 30, 95, 15])  # Fake chances.

# Split and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model.

print("ML model trained with score:", model.score(X_test, y_test))  # Print score, might be low but okay.

# Function to get user input.
def ai_ask(question):
    print(f"AI: {question}")
    return input("You: ")

# Main AI loop.
def run_ai():
    print("AI Assistant starting... Type 'exit' to quit.")
    
    if not user_data:
        print("AI: Hello! I'm an AI with ML to help with unis. Let's create profile.")
        field = ai_ask("Field of interest? (e.g., Computer Science)")
        level = ai_ask("Level? (Bachelor, Master)")
        try:
            gpa = float(ai_ask("GPA? (e.g., 3.5)"))
        except ValueError:
            gpa = 3.0
            print("AI: Invalid, set to 3.0.")
        portfolio = ai_ask("Portfolio description?")
        
        user_data['field'] = field
        user_data['level'] = level
        user_data['gpa'] = gpa
        user_data['portfolio'] = portfolio
        save_data()
        
        print("AI: Profile created!")
    else:
        print("AI: Welcome back!")
    
    while True:
        command = ai_ask("Need? (recommend, chances, apply_help, exit)")
        
        if command.lower() == 'exit':
            break
        elif command.lower() == 'recommend':
            recommendations = recommend_unis(user_data)
            if recommendations:
                print("AI: Recommendations based on analysis:")
                for rec in recommendations:
                    print(rec)
            else:
                print("AI: No matches.")
        elif command.lower() == 'chances':
            chances = estimate_chances(user_data)
            if chances:
                print("AI: ML-estimated chances:")
                for ch in chances:
                    print(ch)
            else:
                print("AI: Recommend first.")
        elif command.lower() == 'apply_help':
            print("AI: Application help:")
            for tip in get_apply_tips():
                print(tip)
        else:
            print("AI: Unknown. Try recommend, chances, apply_help.")

# Recommend function with fuzzy matching.
def recommend_unis(user):
    recs = []
    all_fields = [f for uni in universities.values() for f in uni['fields']]
    
    closest_field = difflib.get_close_matches(user['field'], all_fields, n=1, cutoff=0.6)
    matched_field = closest_field[0] if closest_field else user['field']
    
    for uni, data in universities.items():  # Loop all each time.
        if matched_field in data['fields'] and user['level'] == data['level']:
            if user['gpa'] >= data['avg_gpa'] - 0.3:
                recs.append(f"{uni}: Field {matched_field}, Tuition ${data['tuition']}, Acceptance {data['acceptance_rate']}%")
    
    user['recommended'] = recs
    save_data()
    return recs

# Estimate chances with ML model.
def estimate_chances(user):
    if 'recommended' not in user or not user['recommended']:
        return []
    
    # Extract features from user.
    portfolio_len = len(user['portfolio'].split())
    has_award = 1 if 'award' in user['portfolio'].lower() else 0
    features = np.array([[user['gpa'], portfolio_len, has_award]])
    
    # Predict base chance with ML.
    ml_pred = model.predict(features)[0]
    
    chances = []
    for rec in user['recommended']:
        uni = rec.split(':')[0]
        data = universities[uni]
        # Adjust ML pred with uni data.
        chance = ml_pred * (data['acceptance_rate'] / 10) + random.randint(-5, 5)
        chance = max(0, min(100, chance))
        chances.append(f"{uni}: {chance}% chance (ML estimate)")
    
    return chances

# Apply tips.
def get_apply_tips():
    return [
        "- Gather documents: transcripts, portfolio.",
        "- Strong personal statement.",
        "- Recommendation letters.",
        "- Check deadlines, especially for Early Decision if applying.",
        "- Financial aid if needed.",
        "- Interview practice."
    ]

# Save data.
def save_data():
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(user_data, f)

if __name__ == "__main__":
    run_ai()