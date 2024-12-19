import json
from collections import defaultdict

# Input file path 
input_file_path = "data/train_data.json"
# Output file path
output_file_path = "data/grouped_train_data.json"

# Load the validation dataset
print("Loading validation dataset...")
with open(input_file_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)

# Group questions by context
print("Grouping questions by context...")
grouped_data = defaultdict(list)

for item in validation_data:
    context = item["input"]  
    question = item["output"]  
    grouped_data[context].append(question)

# Format grouped data for output
formatted_data = [
    {
        "context": context,
        "questions": questions
    }
    for context, questions in grouped_data.items()
]

# Save the grouped data to a JSON file
print("Saving grouped data to output file...")
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"Grouped data saved to {output_file_path}")
