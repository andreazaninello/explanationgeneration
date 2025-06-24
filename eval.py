import numpy as np
from typing import List, Dict, Tuple, Optional # Added Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity # Not strictly needed if using F.cosine_similarity
import logging
import json # For reading jsonl files
import os   # For path joining

logger = logging.getLogger(__name__)
# Configure logger if running this file standalone for testing
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EvaluationMetrics:
    """
    Evaluation metrics for explanation generation.
    Implements BLEU-1, ROUGE-1 F1, and a sentence-level BERT-Score F1.
    """
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", device: Optional[str] = None):
        """
        Initialize evaluation metrics.
        
        Args:
            bert_model_name: Name of BERT model to use for BERT-Score calculation.
            device: Optional device string ("cuda", "cpu"). Autodetects if None.
        """
        logger.info(f"Initializing EvaluationMetrics with BERT model: {bert_model_name}")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1 # Common smoothing for BLEU
        
        # Initialize BERT model for BERT-Score
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
        except Exception as e:
            logger.error(f"Failed to load BERT model '{bert_model_name}': {e}")
            logger.error("Please ensure the model name is correct and you have an internet connection or the model is cached.")
            raise
            
        self.bert_model.eval() # Set to evaluation mode
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bert_model.to(self.device)
        logger.info(f"BERT model for evaluation moved to device: {self.device}")
    
    def compute_bleu1(self, predictions: List[str], references: List[List[str]] | List[str]) -> float:
        """
        Compute BLEU-1 score between predictions and references.
        
        Args:
            predictions: List of predicted explanations (strings).
            references: List of reference explanations. Each item can be a single reference string
                        or a list of reference strings (for multiple references per prediction).
            
        Returns:
            Average BLEU-1 score (0-100).
        """
        if not predictions:
            logger.warning("BLEU-1: Empty predictions list. Returning 0.0.")
            return 0.0
        if len(predictions) != len(references):
            raise ValueError(f"BLEU-1: Number of predictions ({len(predictions)}) and references ({len(references)}) must match.")
        
        bleu_scores = []
        
        for i, (pred, ref_item) in enumerate(zip(predictions, references)):
            pred_tokens = pred.lower().split()
            
            # Ensure ref_tokens_list is a list of token lists
            if isinstance(ref_item, str):
                ref_tokens_list = [ref_item.lower().split()]
            elif isinstance(ref_item, list) and all(isinstance(s, str) for s in ref_item):
                ref_tokens_list = [s.lower().split() for s in ref_item]
            else:
                logger.error(f"BLEU-1: Invalid reference format at index {i}. Expected str or List[str], got {type(ref_item)}. Assigning score 0.")
                bleu_scores.append(0.0)
                continue

            if not pred_tokens: # Handle empty prediction
                bleu_scores.append(0.0)
                continue
            # Check if all reference token lists are empty
            if not any(r_list for r_list in ref_tokens_list):
                 bleu_scores.append(0.0)
                 continue

            try:
                score = sentence_bleu(
                    ref_tokens_list, 
                    pred_tokens, 
                    weights=(1.0, 0, 0, 0),  # Only 1-gram for BLEU-1
                    smoothing_function=self.smoothing_function
                )
                bleu_scores.append(score)
            except Exception as e:
                logger.warning(f"BLEU-1 calculation error for pred='{pred}', ref_item='{ref_item}': {e}. Assigning score 0.")
                bleu_scores.append(0.0)
        
        return np.mean(bleu_scores) * 100 if bleu_scores else 0.0
    
    def compute_rouge1(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute ROUGE-1 F1 score between predictions and references.
        
        Args:
            predictions: List of predicted explanations.
            references: List of reference explanations (single string per prediction).
            
        Returns:
            Average ROUGE-1 F1 score (0-100).
        """
        if not predictions:
            logger.warning("ROUGE-1: Empty predictions list. Returning 0.0.")
            return 0.0
        if len(predictions) != len(references):
            raise ValueError(f"ROUGE-1: Number of predictions ({len(predictions)}) and references ({len(references)}) must match.")
        
        rouge_f1_scores = []
        
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip(): # Handle empty or whitespace-only strings
                rouge_f1_scores.append(0.0)
                continue
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_f1_scores.append(scores['rouge1'].fmeasure)
            except Exception as e: # Catch any other errors from rouge_scorer
                logger.warning(f"ROUGE-1 calculation error for pred='{pred}', ref='{ref}': {e}. Assigning score 0.")
                rouge_f1_scores.append(0.0)
        
        return np.mean(rouge_f1_scores) * 100 if rouge_f1_scores else 0.0

    def _get_bert_embeddings(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Helper function to get sentence embeddings using BERT in batches.
        Uses mean pooling of the last hidden state.
        """
        all_embeddings = []
        if not sentences:
            return torch.empty(0, self.bert_model.config.hidden_size).to(self.device)

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Handle cases where a batch might contain only empty or whitespace strings
            processed_batch = [s if s.strip() else "[PAD]" for s in batch_sentences] # Replace empty with [PAD] for tokenizer

            inputs = self.bert_tokenizer(
                processed_batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512 # Standard max_length for BERT
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            last_hidden_states = outputs.last_hidden_state
            
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            mean_pooled_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled_embeddings)
        
        return torch.cat(all_embeddings, dim=0)


    def compute_bert_score_f1(self, predictions: List[str], references: List[str], batch_size: int = 32) -> float:
        """
        Compute a simplified BERT-Score F1 using sentence-level cosine similarity.
        This is an approximation of the original token-level BERTScore.
        
        Args:
            predictions: List of predicted explanations.
            references: List of reference explanations (single string per prediction).
            batch_size: Batch size for processing embeddings.
            
        Returns:
            Average sentence-level cosine similarity (BERT-Score F1 approximation) (0-100).
        """
        if not predictions:
            logger.warning("BERT-Score: Empty predictions list. Returning 0.0.")
            return 0.0
        if len(predictions) != len(references):
            raise ValueError(f"BERT-Score: Number of predictions ({len(predictions)}) and references ({len(references)}) must match.")

        bert_f1_scores = []
        
        try:
            # Get all embeddings in batches
            pred_embeddings = self._get_bert_embeddings(predictions, batch_size=batch_size)
            ref_embeddings = self._get_bert_embeddings(references, batch_size=batch_size)

            if pred_embeddings.shape[0] == 0 or ref_embeddings.shape[0] == 0:
                logger.warning("BERT-Score: Could not generate embeddings for predictions or references (likely all empty). Returning 0.0.")
                return 0.0

            # Compute cosine similarity for all pairs at once
            # (N, D) and (N, D) -> (N)
            similarities = F.cosine_similarity(pred_embeddings, ref_embeddings, dim=1)
            
            # Scale similarities from [-1, 1] to [0, 1] to resemble F1 scores
            scaled_similarities = (similarities + 1) / 2.0
            
            bert_f1_scores = scaled_similarities.cpu().tolist()

        except Exception as e:
            logger.error(f"BERT-Score calculation error during embedding or similarity computation: {e}. May result in partial or zero scores.", exc_info=True)
            # Fallback to individual scoring if batch processing fails, though less ideal
            # For simplicity, if batch fails, we might return 0 or rely on the empty list check.
            # Here, if an error occurs, it's likely systemic; an empty list will lead to 0.0.

        return np.mean(bert_f1_scores) * 100 if bert_f1_scores else 0.0

    def evaluate_batch(self, predictions: List[str], references: List[str | List[str]]) -> Dict[str, float]:
        """
        Computes all configured evaluation metrics for a batch of predictions and references.

        Args:
            predictions: List of predicted explanations.
            references: List of reference explanations. For BLEU, this can be List[List[str]].
                        For ROUGE and BERTScore, it should be List[str].
                        This method adapts by using the first reference if multiple are provided for ROUGE/BERTScore.

        Returns:
            A dictionary containing scores for 'BLEU-1', 'ROUGE-1-F1', 'BERT-Score-F1'.
        """
        if not predictions:
            logger.warning("evaluate_batch: No predictions provided. Returning zero scores.")
            return {
                'BLEU-1': 0.0,
                'ROUGE-1-F1': 0.0,
                'BERT-Score-F1': 0.0
            }

        # Prepare references for ROUGE and BERTScore (expecting List[str])
        simple_references_for_rouge_bert = []
        if references:
            if isinstance(references[0], list): # If references are List[List[str]]
                logger.debug("evaluate_batch: Adapting List[List[str]] references for ROUGE/BERTScore by taking the first reference.")
                simple_references_for_rouge_bert = [ref_list[0] if ref_list else "" for ref_list in references]
            elif isinstance(references[0], str):
                simple_references_for_rouge_bert = references
            else: # Fallback for unexpected format
                logger.warning(f"evaluate_batch: Unexpected reference format: {type(references[0])}. Using empty strings for ROUGE/BERTScore.")
                simple_references_for_rouge_bert = [""] * len(predictions)
        else: # No references provided
            logger.warning("evaluate_batch: No references provided. Using empty strings for ROUGE/BERTScore.")
            simple_references_for_rouge_bert = [""] * len(predictions)
        
        # Ensure simple_references_for_rouge_bert has the same length as predictions
        if len(simple_references_for_rouge_bert) != len(predictions):
            logger.warning(f"Length mismatch after processing references for ROUGE/BERT. Preds: {len(predictions)}, Refs: {len(simple_references_for_rouge_bert)}. Padding refs.")
            simple_references_for_rouge_bert = (simple_references_for_rouge_bert + [""] * len(predictions))[:len(predictions)]


        bleu1_score = self.compute_bleu1(predictions, references) # BLEU handles List[str] or List[List[str]]
        rouge1_score = self.compute_rouge1(predictions, simple_references_for_rouge_bert)
        bert_score_f1 = self.compute_bert_score_f1(predictions, simple_references_for_rouge_bert)

        return {
            'BLEU-1': bleu1_score,
            'ROUGE-1-F1': rouge1_score,
            'BERT-Score-F1': bert_score_f1
        }


def read_jsonl_output(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads a JSONL file containing generated and gold explanations.
    Assumes each line has 'generated_explanation' and 'gold_explanation'.
    """
    predictions = []
    references = []
    if not os.path.exists(file_path):
        logger.warning(f"Output file {file_path} not found. Returning empty lists for evaluation.")
        return predictions, references
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    predictions.append(data.get('generated_explanation', ""))
                    # The 'gold_explanation' corresponds to a single reference string.
                    references.append(data.get('gold_explanation', ""))
                except json.JSONDecodeError:
                    logger.error(f"Skipping malformed JSON line {line_num} in {file_path}: {line.strip()}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

    return predictions, references


if __name__ == '__main__':
    # This block demonstrates how to use the EvaluationMetrics class.
    # It assumes that your main experiment script (e.g., run_experiments.py) 
    # has already run and produced output files.
    
    output_base_dir = "explanation_results" # Should match output_dir in your experiment script
    
    # Use a smaller/faster BERT model for quick local testing of the evaluation script itself
    # For final reported scores, use the intended model like "bert-base-uncased"
    # test_bert_model = "prajjwal1/bert-tiny" 
    test_bert_model = "bert-base-uncased" # Keep as per original for example consistency
    
    try:
        evaluator = EvaluationMetrics(bert_model_name=test_bert_model)
    except Exception as e:
        logger.error(f"Could not initialize EvaluationMetrics: {e}. Exiting evaluation example.")
        exit()

    # Define a list of output files to evaluate
    # This would typically be based on the outputs from your run_experiments.py
    # Format: (Method Description, directory, filename_pattern_or_exact_name)
    eval_tasks = [
        ("Method 1 (Unsup) - Null Prompt", os.path.join("method1_unsupervised"), "null_prompt_generations.jsonl"),
        ("Method 1 (Unsup) - Pattern 0", os.path.join("method1_unsupervised"), "pattern_0_generations.jsonl"),
        # Add other patterns for Method 1 if generated
        ("Method 2 (Finetune) - Null Prompt", os.path.join("method2_finetuning"), "null_prompt_generations.jsonl"),
        ("Method 2 (Finetune) - Pattern 0", os.path.join("method2_finetuning"), "pattern_0_generations.jsonl"),
        # Add other patterns for Method 2 if generated
        ("Method 3 (Ensemble) - Final", os.path.join("method3_ensembling"), "final_ensemble_predictions_generations.jsonl"),
    ]

    all_results_summary = {}

    for desc, method_dir_suffix, filename in eval_tasks:
        full_method_dir = os.path.join(output_base_dir, method_dir_suffix)
        output_file_path = os.path.join(full_method_dir, filename)
        
        logger.info(f"\n--- Evaluating: {desc} ({output_file_path}) ---")
        
        if os.path.exists(output_file_path):
            predictions_list, references_list = read_jsonl_output(output_file_path)
            
            if predictions_list: # Ensure there's data to evaluate
                scores = evaluator.evaluate_batch(predictions_list, references_list)
                logger.info(f"Scores for {desc}:")
                for metric, score_value in scores.items():
                    logger.info(f"  {metric}: {score_value:.2f}")
                all_results_summary[desc] = scores
            else:
                logger.warning(f"No predictions found in {output_file_path} to evaluate, or file was empty/malformed.")
                all_results_summary[desc] = "No data or error"
        else:
            logger.warning(f"Output file not found: {output_file_path}. Skipping evaluation for '{desc}'.")
            all_results_summary[desc] = "File not found"

    print("\n\n--- Overall Evaluation Summary ---")
    for task_desc, result in all_results_summary.items():
        if isinstance(result, dict):
            print(f"\nTask: {task_desc}")
            for metric, score_val in result.items():
                print(f"  {metric}: {score_val:.2f}")
        else:
            print(f"\nTask: {task_desc} - Status: {result}")
    
    # Simple test with dummy data
    print("\n--- Running dummy data test ---")
    dummy_preds = ["this is a test prediction from the evaluation script", "another example here for testing purposes"]
    dummy_refs = ["this is a test reference from the script", "another example for reference testing"] # Single string references
    # dummy_refs_mult = [["this is ref1 script", "this is ref2 script"], ["another example ref1", "another example ref2"]] # For BLEU multi-ref test

    if dummy_preds and dummy_refs:
        dummy_scores = evaluator.evaluate_batch(dummy_preds, dummy_refs)
        print(f"Scores for dummy data (single reference per item):")
        for metric, score in dummy_scores.items():
            print(f"  {metric}: {score:.2f}")
    else:
        print("Dummy predictions or references are empty.")


    print("\nEvaluation script finished.")