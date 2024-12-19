import json
from tqdm import tqdm
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import language_tool_python

# Calculate scores for validation dataset
def calculate_scores(evaluation_results, semantic_model, grammar_tool):
    print("Preparing data for scoring...")
    scores = []
    for result in tqdm(evaluation_results, desc="Scoring contexts"):
        context = result["context"]
        generated_question = result["generated_question"]
        ground_truth_questions = result["ground_truth_questions"]

        max_bleu = 0
        max_rouge1 = 0
        max_rouge2 = 0
        max_rougel = 0
        max_semantic_similarity = 0

        for ground_truth in ground_truth_questions:
            bleu = corpus_bleu([generated_question], [[ground_truth]]).score
            rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_scores = rouge.score(ground_truth, generated_question)
            rouge1 = rouge_scores["rouge1"].fmeasure
            rouge2 = rouge_scores["rouge2"].fmeasure
            rougel = rouge_scores["rougeL"].fmeasure

            embeddings = semantic_model.encode([ground_truth, generated_question])
            semantic_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

            max_bleu = max(max_bleu, bleu)
            max_rouge1 = max(max_rouge1, rouge1)
            max_rouge2 = max(max_rouge2, rouge2)
            max_rougel = max(max_rougel, rougel)
            max_semantic_similarity = max(max_semantic_similarity, semantic_similarity)

        grammar_issues = len(grammar_tool.check(generated_question))

        scores.append({
            "context": context,
            "generated_question": generated_question,
            "ground_truth_questions": ground_truth_questions,
            "max_bleu": max_bleu,
            "max_rouge1": max_rouge1,
            "max_rouge2": max_rouge2,
            "max_rougel": max_rougel,
            "max_semantic_similarity": max_semantic_similarity,
            "grammar_issues": grammar_issues,
        })

    return scores

# Get average scores
def calculate_averages(scores):
    print("Calculating average scores...")
    average_scores = {
        "bleu": sum(score["max_bleu"] for score in scores) / len(scores),
        "rouge1": sum(score["max_rouge1"] for score in scores) / len(scores),
        "rouge2": sum(score["max_rouge2"] for score in scores) / len(scores),
        "rougeL": sum(score["max_rougel"] for score in scores) / len(scores),
        "semantic_similarity": sum(score["max_semantic_similarity"] for score in scores) / len(scores),
        "grammar_issues": sum(score["grammar_issues"] for score in scores) / len(scores),
    }
    return average_scores

def main():
    # Load validation data and results
    print("Loading validation data and evaluation results...")
    with open("results/evaluation_baseline_results.json", "r", encoding="utf-8") as f:
        evaluation_results = json.load(f)

    print("Initializing tools...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    grammar_tool = language_tool_python.LanguageTool('en-US')

    # Calculate scores
    scores = calculate_scores(evaluation_results, semantic_model, grammar_tool)

    # Calculate averages
    average_scores = calculate_averages(scores)

    # Save scores
    print("Saving individual scores and averages...")
    with open("results/individual_baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    with open("results/average_baselin_scores.json", "w", encoding="utf-8") as f:
        json.dump(average_scores, f, indent=2, ensure_ascii=False)

    # Print average scores
    print("Average Scores:")
    for metric, value in average_scores.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
