import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load grouped validation data.
def load_grouped_data(grouped_data_path):
    print("Loading grouped validation dataset...")
    with open(grouped_data_path, "r", encoding="utf-8") as f:
        grouped_data = json.load(f)
    print(f"Loaded {len(grouped_data)} contexts.")
    return grouped_data

# Generate questions for all contexts in batches.
def generate_questions(grouped_data, batch_size, question_generator):
    results = []
    print(f"Generating questions for {len(grouped_data)} contexts in batches of {batch_size}...")

    for i in tqdm(range(0, len(grouped_data), batch_size), desc="Processing batches"):
        batch_contexts = grouped_data[i:i + batch_size]

        input_texts = [
            f"Generate a question based on the context: {item['context']}" for item in batch_contexts
        ]

        batch_generated_questions = question_generator(input_texts, max_length=50)

        for item, output in zip(batch_contexts, batch_generated_questions):
            results.append({
                "context": item["context"],
                "ground_truth_questions": item["questions"],
                "generated_question": output["generated_text"]
            })

    return results

# Save generated questions to a JSON file.
def save_results(results, output_path):
    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}.")

def main():
    grouped_data_path = "data/grouped_validation_data.json"  
    baseline_output_path = "data/evaluation_baseline_results.json"
    baseline_model_path = "google/flan-t5-base"  
    batch_size = 20  

    grouped_data = load_grouped_data(grouped_data_path)

    print("Evaluating with baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_path)
    question_generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0  
    )
    baseline_results = generate_questions(grouped_data, batch_size, question_generator)

    save_results(baseline_results, baseline_output_path)

if __name__ == "__main__":
    main()
