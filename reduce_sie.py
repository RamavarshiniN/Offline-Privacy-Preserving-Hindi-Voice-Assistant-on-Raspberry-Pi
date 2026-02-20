import fasttext
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "brain_model.bin")

# Load existing model
print("📦 Loading model...")
model = fasttext.load_model(model_path)

# Compress
print("⚙ Compressing model...")
model.quantize(
    input=None,      
    qnorm=True,
    retrain=False
)

compressed_path = os.path.join(current_dir, "brain_model.ftz")
model.save_model(compressed_path)

print(f"✅ Compressed model saved at: {compressed_path}")
