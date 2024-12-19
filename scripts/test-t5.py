import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model_path = ".models/fine_tuned_flan_t5_full"  # Replace with your saved model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Ensure the model runs on the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a sample context and test question generation
context = (
    "On July 20, 1969, NASA's Apollo 11 mission successfully landed the first humans on the Moon. Astronauts Neil Armstrong and Buzz Aldrin spent approximately two and a half hours outside the spacecraft, while Michael Collins remained in lunar orbit aboard the command module. This historic event marked a significant achievement in space exploration and fulfilled President John F. Kennedy's goal of landing a man on the Moon before the end of the 1960s."
)

# Tokenize the input
input_text = f"Generate a question based on the following context: {context}"
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

# Generate the output
output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated question
print("Generated Question:", generated_text)
