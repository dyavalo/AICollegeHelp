# simple ai helper for choosing colleges
# i added sklearn to make it look like real ml but the dataset is tiny and made up
# code isnt perfect, sometimes crashes if you type wrong but it works mostly

import difflib
import random
import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np

DATA_FILE = 'my_data.pkl'

unis = {
    'Harvard': {'fields': ['CS', 'Business', 'Medicine'], 'level': 'Bachelor', 'gpa': 3.9, 'tuition': 50000, 'rate': 5},
    'MIT': {'fields': ['Engineering', 'CS'], 'level': 'Bachelor', 'gpa': 3.95, 'tuition': 55000, 'rate': 7},
    'Stanford': {'fields': ['Business', 'CS'], 'level': 'Bachelor', 'gpa': 3.9, 'tuition': 52000, 'rate': 4},
    'BU': {'fields': ['Business', 'CS'], 'level': 'Bachelor', 'gpa': 3.7, 'tuition': 58000, 'rate': 18},
}

# fake data for training
X = np.array([[3.5, 30, 0], [3.8, 80, 1], [4.0, 100, 1], [3.2, 20, 0]])
y = np.array([30, 70, 90, 15])

model = LinearRegression()
model.fit(X, y)

user = {}

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        user = pickle.load(f)

def ask(q):
    print("AI:", q)
    return input("You: ")

if not user:
    print("hey lets make your profile")
    user['field'] = ask("What major? (like CS)")
    user['level'] = ask("Level? (Bachelor)")
    try:
        user['gpa'] = float(ask("Your GPA?"))
    except:
        user['gpa'] = 3.0
        print("AI: didnt understand, set to 3.0")
    user['portfolio'] = ask("Describe your portfolio (projects, awards etc)")
    
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(user, f)

while True:
    cmd = ask("What do you want? recommend, chances, tips, exit")
    if cmd.lower() == 'exit':
        break
    elif cmd.lower() == 'recommend':
        recs = []
        close = difflib.get_close_matches(user['field'], [f for u in unis.values() for f in u['fields']], 1, 0.5)
        mf = close[0] if close else user['field']
        for name, d in unis.items():
            if mf in d['fields'] and user['level'] == d['level']:
                if user['gpa'] > d['gpa'] - 0.4:
                    recs.append(f"{name} - tuition ${d['tuition']}, acceptance {d['rate']}%")
        print("Recommendations:\n" + "\n".join(recs) if recs else "nothing found :(")
    elif cmd.lower() == 'chances':
        plen = len(user['portfolio'].split())
        award = 1 if 'award' in user['portfolio'].lower() else 0
        pred = model.predict(np.array([[user['gpa'], plen, award]]))[0]
        print(f"Chance around {int(pred + random.randint(-10,10))}% according to ml (but data is fake)")
    elif cmd.lower() == 'tips':
        print("Tips: get your transcripts, write a good essay about yourself, ask teachers for recommendations, check deadlines")
