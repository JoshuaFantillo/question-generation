# Question Generation Project

## Description
This project fine-tunes a T5 model for text-to-question generation. The workflow includes data preprocessing, model training, evaluation, and custom question generation from user-provided files (e.g., `.txt` or `.pptx`).

---

## Installation

### Install required dependencies:
```bash
pip install -r requirements.txt
```
### How to Run the Project

#### 1. Data Preprocessing
The `data-preprocessing.py` script formats raw data into the required structure for T5 fine-tuning.

**Command to run:**
```bash
python scripts/data-preprocessing.py
```
**Required Files:**
- The script automatically downloads the SQuAD dataset using Hugging Face's library.

**Outputs:**
- Preprocessed training and validation files in the `data/` directory:
  - `data/train_data.json`
  - `data/validation_data.json`
### 2. Grouping Questions by Context

The `combine-data.py` script groups questions by their shared context to enhance model evaluation.

**Command to run:**
```bash
python scripts/combine-data.py
```
**Required Files:**

- **Input file:** `data/train_data.json` (generated from the preprocessing step).

**Outputs:**

- **Output file:** `data/grouped_train_data.json`.
### 3. Fine-Tuning the Model

The `finetune-t5.py` script fine-tunes the T5 model using the grouped dataset.

**Command to run:**

```bash
python scripts/finetune-t5.py
```
**Required Files:**

- **Input files:**
  - `data/grouped_train_data.json`
  - `data/grouped_validation_data.json`

**Outputs:**

- Fine-tuned model saved in the `model_output/` directory.
### 4. Evaluating the Model

The `evaluate-t5.py` script generates questions and compares them to the ground truth.

**Command to run:**

```bash
python scripts/evaluate-t5.py
```
**Required Files:**

- **Input file:**
  - `data/grouped_validation_data.json`

**Outputs:**

- `results/evaluation_baseline_results.json`
### 5. Generating Scores

The `get-scores.py` script calculates evaluation metrics such as BLEU, ROUGE, semantic similarity, and grammar correctness.

**Command to run:**

```bash
python scripts/get-scores.py
```
**Required Files:**

- **Input file:**
  - `results/evaluation_baseline_results.json`

**Outputs:**

- Individual scores: `results/individual_baseline_scores.json`
- Average scores: `results/average_baseline_scores.json`

### 6. Testing Custom Inputs

The `input.py` script generates questions from user-provided `.txt` or `.pptx` files.

**Command to run:**

```bash
python input.py --input_path <file_path> --num_questions <number_of_questions> --output <output_path>
```
**Required Arguments:**

- `--input_path`: Path to the `.txt` or `.pptx` file containing your custom data.
- `--num_questions`: Number of questions to generate for each context (default: 1).
- `--output`: Path to save the output JSON file.

## Directory Structure

```plaintext
project/
│
├── data/
│   ├── train_data.json
│   ├── validation_data.json
│   ├── grouped_train_data.json
│   ├── grouped_validation_data.json
│
├── results/
│   ├── evaluation_baseline_results.json
│   ├── individual_baseline_scores.json
│   ├── average_baseline_scores.json
│
├── model_output/
│   ├── [Fine-tuned model files]
│
├── scripts/
│   ├── data-preprocessing.py
│   ├── combine-data.py
│   ├── finetune-t5.py
│   ├── evaluate-t5.py
│   ├── get-scores.py
│   ├── input.py
│
├── requirements.txt
├── README.md
└── ...
```
## Notes

- Ensure the `data/` directory exists before running scripts that generate outputs.
- Make sure you have sufficient GPU resources for fine-tuning (`finetune-t5.py`) and evaluation.
- Use `python input.py --help` for additional options when testing custom inputs.

## References

1. Asahi Ushio, Fernando Alva Manchego, and Jose Camacho-Collados, *Generative Language Models for Paragraph-Level Question Generation*.

2. Sahan Bulathwela, Hamze Muse, and Emine Yilmaz, *Scalable Educational Question Generation with Pre-trained Language Models*.

3. Bidyut Das, Mukta Majumder, Santanu Phadikar, and Arif Ahmed Sekh, *Automatic Question Generation and Answer Assessment: A Survey*.
