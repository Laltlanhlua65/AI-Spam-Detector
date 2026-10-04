from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

messages = [
    "Win free money lottery prize", "Hi meeting tomorrow at 9am",
    "Congratulations you won free prize money", "Please submit assignment today",
    "Free cash prize click here to claim", "Can we meet for lunch",
    "Urgent claim your free reward now", "Project report is ready",
    "You have won 1000 dollars prize", "Lets go for tea at 4pm",
    "Get cheap meds free now", "Are you coming to class",
    "You are selected for free gift prize", "Notes have been shared in group",
    "Earn money fast online free", "See you at library tomorrow",
    "Lottery winner claim money now", "Assignment deadline is extended",
    "Free iphone click here to win", "Dinner at my place tonight",
    "Claim your cash prize free money", "Lab report submitted successfully",
    "Free vacation prize winner", "Bus is late today",
    "You won free tickets prize", "Exam postponed to next week",
    "Urgent money transfer free prize", "Can you send the pdf",
    "Win big lottery prize money", "Group study at 5pm in library",
    "Free bitcoin win money now", "Class notes are ready",
    "Congratulations free cash won", "Tomorrow is holiday",
    "Earn 5000 per week free", "Submit your lab manual",
    "You won jackpot prize money", "See you in college",
    "Click here for free prize", "What is homework today"
]
labels = [1,0]*20 # 40 labels for 40 messages

X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.25, random_state=1)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectors, y_train)

prediction = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, prediction)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("---------------------------")

while True:
    msg = input("\nEnter Email/Message (type 'exit' to quit): ")
    if msg.lower() == 'exit':
        break
    msg_vector = vectorizer.transform([msg])
    result = model.predict(msg_vector)
    if result[0] == 1:
        print("Result: SPAM Detected")
    else:
        print("Result: HAM - Not Spam")