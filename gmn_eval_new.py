import numpy as np
from typing import List, Dict, Tuple, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import logging
import json
import os
import csv # For writing CSV results

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EvaluationMetrics:
    """
    Evaluation metrics for explanation generation.
    Implements BLEU-1, ROUGE-1 F1, and a sentence-level BERT-Score F1.
    """
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", device: Optional[str] = None):
        logger.info(f"Initializing EvaluationMetrics with BERT model: {bert_model_name}")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
        except Exception as e:
            logger.error(f"Failed to load BERT model '{bert_model_name}': {e}", exc_info=True)
            raise
            
        self.bert_model.eval()
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bert_model.to(self.device)
        logger.info(f"BERT model for evaluation moved to device: {self.device}")
    
    def compute_bleu1(self, predictions: List[str], references: List[List[str]] | List[str]) -> float:
        if not predictions: return 0.0
        if len(predictions) != len(references):
            logger.error(f"BLEU-1: Mismatch len predictions ({len(predictions)}) vs references ({len(references)}).")
            # Pad shorter list with empty items to attempt scoring, or raise error
            # For simplicity here, we might just return 0 or score up to the shorter length.
            # However, the calling evaluate_batch should ideally ensure lengths match.
            return 0.0 # Or handle more gracefully
        
        bleu_scores = []
        for i, (pred, ref_item) in enumerate(zip(predictions, references)):
            pred_tokens = pred.lower().split()
            if isinstance(ref_item, str): ref_tokens_list = [ref_item.lower().split()]
            elif isinstance(ref_item, list) and all(isinstance(s, str) for s in ref_item): ref_tokens_list = [s.lower().split() for s in ref_item]
            else:
                logger.error(f"BLEU-1: Invalid ref format at index {i}. Assigning score 0.")
                bleu_scores.append(0.0); continue
            if not pred_tokens or not any(r_list for r_list in ref_tokens_list):
                bleu_scores.append(0.0); continue
            try:
                score = sentence_bleu(ref_tokens_list, pred_tokens, weights=(1.0,0,0,0), smoothing_function=self.smoothing_function)
                bleu_scores.append(score)
            except Exception as e:
                logger.warning(f"BLEU-1 calc error for pred='{pred[:30]}...', ref='{str(ref_item)[:30]}...': {e}. Score 0.")
                bleu_scores.append(0.0)
        return np.mean(bleu_scores) * 100 if bleu_scores else 0.0
    
    def compute_rouge1(self, predictions: List[str], references: List[str]) -> float:
        if not predictions: return 0.0
        if len(predictions) != len(references):
            logger.error(f"ROUGE-1: Mismatch len predictions ({len(predictions)}) vs references ({len(references)}).")
            return 0.0
        
        rouge_f1_scores = []
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip(): rouge_f1_scores.append(0.0); continue
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_f1_scores.append(scores['rouge1'].fmeasure)
            except Exception as e:
                logger.warning(f"ROUGE-1 calc error for pred='{pred[:30]}...', ref='{ref[:30]}...': {e}. Score 0.")
                rouge_f1_scores.append(0.0)
        return np.mean(rouge_f1_scores) * 100 if rouge_f1_scores else 0.0

    def _get_bert_embeddings(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        if not sentences: return torch.empty(0, self.bert_model.config.hidden_size).to(self.device)
        for i in range(0, len(sentences), batch_size):
            batch_sentences = [s if s.strip() else "[PAD]" for s in sentences[i:i+batch_size]]
            inputs = self.bert_tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad(): outputs = self.bert_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled_embeddings)
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.bert_model.config.hidden_size).to(self.device)

    def compute_bert_score_f1(self, predictions: List[str], references: List[str], batch_size: int = 32) -> float:
        if not predictions: return 0.0
        if len(predictions) != len(references):
            logger.error(f"BERT-Score: Mismatch len predictions ({len(predictions)}) vs references ({len(references)}).")
            return 0.0
        
        bert_f1_scores = []
        try:
            pred_embeddings = self._get_bert_embeddings(predictions, batch_size=batch_size)
            ref_embeddings = self._get_bert_embeddings(references, batch_size=batch_size)
            if pred_embeddings.shape[0] == 0 or ref_embeddings.shape[0] == 0: return 0.0
            similarities = F.cosine_similarity(pred_embeddings, ref_embeddings, dim=1)
            scaled_similarities = (similarities + 1) / 2.0
            bert_f1_scores = scaled_similarities.cpu().tolist()
        except Exception as e:
            logger.error(f"BERT-Score calc error: {e}", exc_info=True)
        return np.mean(bert_f1_scores) * 100 if bert_f1_scores else 0.0

    def evaluate_batch(self, predictions: List[str], references: List[str | List[str]]) -> Dict[str, float]:
        if not predictions:
            return {'BLEU-1': 0.0, 'ROUGE-1-F1': 0.0, 'BERT-Score-F1': 0.0}

        simple_refs = []
        if references:
            if isinstance(references[0], list): simple_refs = [r[0] if r else "" for r in references]
            elif isinstance(references[0], str): simple_refs = references
            else: simple_refs = [""] * len(predictions)
        else: simple_refs = [""] * len(predictions)
        
        # Ensure lengths match for ROUGE/BERT by padding/truncating if necessary
        # This is a fallback; ideally, the input should be correct.
        if len(simple_refs) < len(predictions):
            simple_refs.extend([""] * (len(predictions) - len(simple_refs)))
        elif len(simple_refs) > len(predictions):
            simple_refs = simple_refs[:len(predictions)]
            
        if len(references) < len(predictions): # For BLEU, if original refs are too short
             bleu_references = (list(references) + [[""]] * (len(predictions) - len(references)))
        elif len(references) > len(predictions):
             bleu_references = list(references)[:len(predictions)]
        else:
             bleu_references = references


        bleu1_score = self.compute_bleu1(predictions, bleu_references)
        rouge1_score = self.compute_rouge1(predictions, simple_refs)
        bert_score_f1 = self.compute_bert_score_f1(predictions, simple_refs)

        return {'BLEU-1': bleu1_score, 'ROUGE-1-F1': rouge1_score, 'BERT-Score-F1': bert_score_f1}

def read_jsonl_output(file_path: str) -> Tuple[List[str], List[str]]:
    predictions, references = [], []
    if not os.path.exists(file_path):
        logger.warning(f"Output file {file_path} not found.")
        return predictions, references
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    predictions.append(data.get('generated_explanation', ""))
                    references.append(data.get('gold_explanation', ""))
                except json.JSONDecodeError:
                    logger.error(f"Skipping malformed JSON line {line_num} in {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return predictions, references

def write_results_to_csv(all_results_summary: Dict, csv_filepath: str):
    """Writes the evaluation summary to a CSV file."""
    fieldnames = ['Model', 'Method', 'Pattern', 'BLEU-1', 'ROUGE-1-F1', 'BERT-Score-F1', 'Status']
    
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_name, model_data in all_results_summary.items():
            if isinstance(model_data, str): # Error for the whole model
                writer.writerow({'Model': model_name, 'Status': model_data})
                continue
            for method_pattern_key, scores_or_status in model_data.items():
                # method_pattern_key is like "Method 1 (Unsup) - Null Prompt"
                parts = method_pattern_key.split(" - ")
                method_desc = parts[0]
                pattern_desc = parts[1] if len(parts) > 1 else "N/A"
                
                row = {'Model': model_name, 'Method': method_desc, 'Pattern': pattern_desc}
                if isinstance(scores_or_status, dict):
                    row['BLEU-1'] = f"{scores_or_status.get('BLEU-1', 0.0):.2f}"
                    row['ROUGE-1-F1'] = f"{scores_or_status.get('ROUGE-1-F1', 0.0):.2f}"
                    row['BERT-Score-F1'] = f"{scores_or_status.get('BERT-Score-F1', 0.0):.2f}"
                    row['Status'] = 'Success'
                else: # It's a status string like "File not found"
                    row['Status'] = scores_or_status
                writer.writerow(row)
    logger.info(f"Evaluation summary written to {csv_filepath}")


if __name__ == '__main__':
    # Base directory where model-specific results are stored
    # This should match 'base_output_directory' from your experiment script
    results_base_dir = "generations" # Or whatever you used, e.g., "explanation_results_multi_model"
    
    # BERT model for evaluation metrics
    eval_bert_model = "bert-base-uncased" 
    
    try:
        evaluator = EvaluationMetrics(bert_model_name=eval_bert_model)
    except Exception as e:
        logger.error(f"Could not initialize EvaluationMetrics: {e}. Exiting evaluation.")
        exit()

    # Define models short names as used in directory creation by the experiment script
    # THIS LIST MUST MATCH THE 'short_name's FROM YOUR EXPERIMENT SCRIPT'S models_to_run_config
    model_short_names = ["bart-base", "t5-base", "pegasus-xsum"] 

    # Define standard method directories and generation files
    method_configs = {
        "method1_unsupervised": [
            ("Null Prompt", "null_prompt_generations.jsonl"),
            ("Pattern 0", "pattern_0_generations.jsonl"),
            ("Pattern 1", "pattern_1_generations.jsonl"),
            ("Pattern 2", "pattern_2_generations.jsonl"),
            ("Pattern 3", "pattern_3_generations.jsonl"),
        ],
        "method2_finetuning": [
            ("Null Prompt", "null_prompt_generations.jsonl"),
            ("Pattern 0", "pattern_0_generations.jsonl"),
            ("Pattern 1", "pattern_1_generations.jsonl"),
            ("Pattern 2", "pattern_2_generations.jsonl"),
            ("Pattern 3", "pattern_3_generations.jsonl"),
        ],
        "method3_ensembling": [
            ("Final Ensemble", "final_ensemble_predictions_generations.jsonl"),
        ]
    }

    overall_summary_all_models = {}

    for model_sname in model_short_names:
        model_dir_path = os.path.join(results_base_dir, model_sname)
        logger.info(f"\n\n{'='*15} EVALUATING MODEL: {model_sname} (from {model_dir_path}) {'='*15}")
        
        if not os.path.isdir(model_dir_path):
            logger.warning(f"Directory for model {model_sname} not found: {model_dir_path}. Skipping.")
            overall_summary_all_models[model_sname] = f"Directory not found: {model_dir_path}"
            continue

        model_specific_summary = {}
        for method_dir_name, patterns_and_files in method_configs.items():
            method_path = os.path.join(model_dir_path, method_dir_name)
            
            if not os.path.isdir(method_path):
                logger.debug(f"Method directory not found: {method_path}. Assuming no results for this method for {model_sname}.")
                continue 

            for pattern_desc, filename in patterns_and_files:
                output_file_path = os.path.join(method_path, filename)
                
                eval_key = f"{method_dir_name.replace('_', ' ').title()} - {pattern_desc}"
                logger.info(f"\n--- Evaluating: {eval_key} (File: {output_file_path}) ---")
                
                if os.path.exists(output_file_path):
                    predictions_list, references_list = read_jsonl_output(output_file_path)
                    
                    if predictions_list:
                        scores = evaluator.evaluate_batch(predictions_list, references_list)
                        logger.info(f"Scores for {eval_key}:")
                        for metric, score_value in scores.items():
                            logger.info(f"  {metric}: {score_value:.2f}")
                        model_specific_summary[eval_key] = scores
                    else:
                        status_msg = f"No predictions in {output_file_path} or file empty/malformed."
                        logger.warning(status_msg)
                        model_specific_summary[eval_key] = status_msg
                else:
                    status_msg = f"Output file not found: {output_file_path}."
                    logger.debug(status_msg)
                    model_specific_summary[eval_key] = status_msg
        
        overall_summary_all_models[model_sname] = model_specific_summary

    # --- Print Overall Evaluation Summary ---
    print("\n\n" + "="*20 + " OVERALL EVALUATION SUMMARY " + "="*20)
    for model_name, model_results in overall_summary_all_models.items():
        print(f"\n\nMODEL: {model_name.upper()}")
        if isinstance(model_results, str):
            print(f"  Status: {model_results}")
            continue
        if not model_results:
            print("  No results found or evaluated for this model.")
            continue
            
        for task_desc, result in model_results.items():
            if isinstance(result, dict):
                print(f"\n  Task: {task_desc}")
                for metric, score_val in result.items():
                    print(f"    {metric}: {score_val:.2f}")
            else:
                print(f"\n  Task: {task_desc} - Status: {result}")
    
    # --- Write results to CSV ---
    csv_output_filepath = os.path.join(results_base_dir, "evaluation_summary_all_models.csv")
    write_results_to_csv(overall_summary_all_models, csv_output_filepath)
    
    print("\nEvaluation script finished.")