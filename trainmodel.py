import fasttext
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_dir, "training.txt")
model_path = os.path.join(current_dir, "brain_model.bin")

print(f"📂 Looking for training data at: {train_file}")

# 3. Check if file exists before crashing
if not os.path.exists(train_file):
    print("❌ ERROR: training.txt not found!")
    print("👉 Please make sure 'training.txt' is inside the 'fast' folder.")
    exit(1)

# 4. Train the model
model = fasttext.train_supervised(
    input=train_file,
    lr=1.0,
    epoch=25,
    wordNgrams=2,
    bucket=200000,
    dim=50,
    loss='softmax',
    thread=4
)

# 5. Save the model
model.save_model(model_path)
print(f"✅ Model saved successfully at: {model_path}")

# Quick test
def predict_intent(text):
    labels, probabilities = model.predict(text)
    return labels[0], probabilities[0]

test_sentence = "फैन चालू करो"
label, prob = predict_intent(test_sentence)
print(f"Sentence: {test_sentence}\nPredicted Label: {label}, Probability: {prob:.4f}")
