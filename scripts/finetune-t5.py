import json
from functools import partial
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

# Preprocess the dataset
def preprocess_data(item):
    if "context" in item and "questions" in item:
        context = item["context"]
        if len(item["questions"]) > 0:
            question = item["questions"][0]  
            return {
                "input_text": f"Generate a question based on the context: {context}",
                "target_text": question,
            }
    return None

# Load a JSON file and convert it to a Hugging Face Dataset
def load_json_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Tokenize the data
def tokenize_data(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        batch["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

# Used for debugging
def debug_dataset(dataset, name):
    """Print debugging information about the dataset."""
    print(f"\n=== Debugging {name} Dataset ===")
    print(f"Total examples: {len(dataset)}")
    print("Sample entry:")
    for key, value in dataset[0].items():
        print(f"  {key}: {value}")
    print("===")


def main():
    train_path = "data/grouped_train_data.json"
    validation_path = "data/grouped_validation_data.json"
    output_dir = "./fine_tuned_flan_t5_full"
    model_name = "google/flan-t5-base"
    batch_size = 8
    max_examples = 100  # Used for testing

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Loading grouped dataset...")
    train_data = load_json_dataset(train_path)
    validation_data = load_json_dataset(validation_path)

    print("Preprocessing datasets...")
    preprocess = partial(preprocess_data)
    train_data = train_data.map(preprocess).filter(lambda x: x is not None)
    validation_data = validation_data.map(preprocess).filter(lambda x: x is not None)

    # Limit datasets to max_examples for debugging
    #train_data = train_data.select(range(min(len(train_data), max_examples)))
    #validation_data = validation_data.select(range(min(len(validation_data), max_examples)))

    print("Tokenizing datasets...")
    tokenize_with_tokenizer = partial(tokenize_data, tokenizer=tokenizer)
    train_data = train_data.map(tokenize_with_tokenizer, batched=True)
    validation_data = validation_data.map(tokenize_with_tokenizer, batched=True)

    # Debugging tokenized datasets
    debug_dataset(train_data, "Train")
    debug_dataset(validation_data, "Validation")

    # Finetuning arguments 
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        bf16=True,
        gradient_accumulation_steps=4, 
        logging_dir="./logs",
        logging_steps=50,
        max_grad_norm=0.5, 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
