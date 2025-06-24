import torch
import torch.nn as nn
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration,
    Adafactor
)
from torch.utils.data import Dataset, DataLoader
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
# from sklearn.metrics import accuracy_score # Not used
import re
import os
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExplanationExample:
    """Data structure for e-SNLI examples"""
    premise: str
    hypothesis: str
    label: str
    explanation: str
    guid: str = ""

class PromptTemplate:
    """Class to handle prompt templates as described in the paper"""
    
    def __init__(self):
        self.prefixes = ["Explanation:", "Rationale:"]
        self.verbalizers = {
            "ENTAILMENT": "entails",
            "CONTRADICTION": "contradicts", 
            "NEUTRAL": "does not entail"
        }
        
    def create_patterns(self) -> List[str]: # Not actively used but defined as per original
        patterns = []
        patterns.append("patt1")
        patterns.append("patt2")
        patterns.append("patt3")
        patterns.append("patt4")
        return patterns
    
    def format_input(self, example: ExplanationExample, pattern_id: int, 
                    prefix_id: int, mask_token: str = "<mask>") -> str:
        premise = example.premise.strip()
        hypothesis = example.hypothesis.strip()
        # Fallback to the label itself if not in verbalizers (e.g. if labels are already verbs)
        verbalizer = self.verbalizers.get(example.label.upper(), example.label) 
        prefix = self.prefixes[prefix_id]
        
        if pattern_id in [0, 2]: # patt1, patt3
            return f"{premise} {verbalizer} {hypothesis} {prefix} {mask_token}"
        else: # patt2, patt4
            return f"{prefix} {mask_token} Text: {premise} {verbalizer} {hypothesis}"
    
    def format_target(self, example: ExplanationExample, pattern_id: int,
                     prefix_id: int, include_prefix: bool = True) -> str:
        prefix = self.prefixes[prefix_id] if include_prefix else ""
        explanation = example.explanation.strip()
        
        if pattern_id in [0, 2] and include_prefix:
            return f"{prefix} {explanation}"
        else:
            return explanation

class ESNLIDataset(Dataset):
    """Dataset class for e-SNLI data with prompting"""
    
    def __init__(self, examples: List[ExplanationExample], tokenizer,
                 prompt_template: PromptTemplate, max_length: int = 512,
                 pattern_ids: List[int] = None, prefix_ids: List[int] = None,
                 is_training: bool = True):
        
        self.examples = examples
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.is_training = is_training
        
        self.pattern_ids = pattern_ids if pattern_ids else [0, 1, 2, 3]
        self.prefix_ids = prefix_ids if prefix_ids else [0, 1]
        
        self.formatted_examples = []
        if not examples: # Handle empty examples list
            logger.warning("ESNLIDataset initialized with no examples.")
        for example in examples:
            for pattern_id in self.pattern_ids:
                for prefix_id in self.prefix_ids:
                    self.formatted_examples.append((example, pattern_id, prefix_id))
    
    def __len__(self):
        return len(self.formatted_examples)
    
    def __getitem__(self, idx):
        if idx >= len(self.formatted_examples):
            raise IndexError("Index out of bounds for ESNLIDataset")
        example, pattern_id, prefix_id = self.formatted_examples[idx]
        
        # Ensure mask_token is available
        mask_token_str = self.tokenizer.mask_token if self.tokenizer.mask_token else "<mask>"

        input_text = self.prompt_template.format_input(
            example, pattern_id, prefix_id, mask_token_str
        )
        
        target_text = self.prompt_template.format_target(
            example, pattern_id, prefix_id, include_prefix=True
        )
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0), # Squeeze batch dim
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0), # Not typically used for labels
            'pattern_id': pattern_id,
            'prefix_id': prefix_id,
            'original_example': example
        }

def custom_collate_fn(batch_list: List[Dict]) -> Dict:
    collated_batch = {}
    if not batch_list:
        return collated_batch
        
    first_item_keys = batch_list[0].keys()
    
    for key in first_item_keys:
        elements = [item[key] for item in batch_list]
        if key == 'original_example':
            collated_batch[key] = elements
        elif isinstance(elements[0], torch.Tensor):
            collated_batch[key] = torch.stack(elements)
        elif isinstance(elements[0], (int, float)):
            collated_batch[key] = torch.tensor(elements, dtype=torch.long)
        else: # Fallback for other types
            try:
                collated_batch[key] = torch.utils.data._utils.collate.default_collate(elements)
            except TypeError: # If default_collate fails, store as list
                collated_batch[key] = elements 
                
    return collated_batch

class ExplanationGenerator:
    """Main class implementing the three methods from the paper"""
    
    def __init__(self, model_name: str = "facebook/bart-large", 
                 device: str = "cuda", output_dir: str = "results"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ExplanationGenerator for {self.model_name} using device: {self.device}")
        logger.info(f"Results will be saved in: {self.output_dir}")
        self.prompt_template = PromptTemplate()
        
        self._load_model_and_tokenizer() # Initialize model and tokenizer
        
    def _load_model_and_tokenizer(self):
        logger.info(f"Loading model and tokenizer for {self.model_name}...")
        model_name_lower = self.model_name.lower()
        if "bart" in model_name_lower:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        elif "t5" in model_name_lower:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        elif "pegasus" in model_name_lower:
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model type in: {self.model_name}")
        
        self.model.to(self.device)
        
        # Ensure mask_token is '<mask>' as per paper's prompts
        # This standardizes the mask token for prompt formatting.
        desired_mask_token_str = "<mask>"
        if self.tokenizer.mask_token != desired_mask_token_str:
            if desired_mask_token_str not in self.tokenizer.get_vocab():
                logger.info(f"Adding '{desired_mask_token_str}' to tokenizer vocab and resizing model embeddings.")
                self.tokenizer.add_special_tokens({'mask_token': desired_mask_token_str})
                self.model.resize_token_embeddings(len(self.tokenizer))
            else:
                logger.info(f"Setting tokenizer.mask_token to '{desired_mask_token_str}'.")
            self.tokenizer.mask_token = desired_mask_token_str # Set it as the active mask_token

        if self.tokenizer.pad_token is None:
            # Common practice: use eos_token as pad_token if not defined
            if self.tokenizer.eos_token:
                logger.info(f"Setting pad_token to eos_token: {self.tokenizer.eos_token}")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else: # Add a generic pad token if eos is also missing
                logger.info("Adding a generic [PAD] token as pad_token.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model and tokenizer for {self.model_name} loaded. Mask token: '{self.tokenizer.mask_token}', Pad token: '{self.tokenizer.pad_token}'")

    def _get_mask_token(self) -> str:
        """Centralized way to get the mask token string."""
        return self.tokenizer.mask_token if self.tokenizer.mask_token else "<mask>"


    def _write_generations_to_file(self,
                                   test_examples: List[ExplanationExample],
                                   generated_explanations: List[str],
                                   method_name: str,
                                   pattern_name: str = "default_pattern"):
        safe_method_name = re.sub(r'\W+', '_', method_name)
        safe_pattern_name = re.sub(r'\W+', '_', pattern_name)

        method_output_dir = os.path.join(self.output_dir, safe_method_name)
        os.makedirs(method_output_dir, exist_ok=True)
        
        output_filename = f"{safe_pattern_name}_generations.jsonl"
        output_path = os.path.join(method_output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if len(test_examples) != len(generated_explanations):
                logger.warning(f"Mismatch lengths: {len(test_examples)} examples vs {len(generated_explanations)} explanations for {method_name}/{pattern_name}.")
            
            for ex, gen_expl in zip(test_examples, generated_explanations):
                record = {
                    "guid": ex.guid, "premise": ex.premise, "hypothesis": ex.hypothesis,
                    "gold_label": ex.label, "gold_explanation": ex.explanation,
                    "generated_explanation": gen_expl, "method": method_name, "pattern": pattern_name
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved generations to {output_path}")

    def _extract_explanation(self, generated_text: str, input_text: str) -> str:
        mask_token_str = self._get_mask_token()
        explanation = generated_text.strip()
        
        prompt_prefix = input_text.split(mask_token_str, 1)[0]
        if explanation.startswith(prompt_prefix.strip()):
            explanation = explanation[len(prompt_prefix.strip()):].strip()
        elif explanation.startswith(prompt_prefix):
            explanation = explanation[len(prompt_prefix):].strip()

        for pfx_to_remove in self.prompt_template.prefixes:
            if explanation.lower().startswith(pfx_to_remove.lower()):
                explanation = explanation[len(pfx_to_remove):].strip()
        return explanation

    def method1_unsupervised(self, test_examples: List[ExplanationExample],
                           pattern_ids: List[int] = None) -> Dict[str, List[str]]:
        method_name_str = "method1_unsupervised"
        logger.info(f"Running {method_name_str} for {self.model_name}")
        self.model.eval()
        results = {}
        mask_token_str = self._get_mask_token()
        
        pattern_ids_to_run = pattern_ids or [0, 1, 2, 3]
        
        for pattern_id in pattern_ids_to_run:
            pattern_name_str = f"pattern_{pattern_id}"
            pattern_results = []
            logger.info(f"{method_name_str}: Generating with {pattern_name_str}")
            for example in test_examples:
                input_text = self.prompt_template.format_input(example, pattern_id, 0, mask_token_str)
                with torch.no_grad():
                    inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    explanation = self._extract_explanation(generated_text, input_text)
                    pattern_results.append(explanation)
            results[pattern_name_str] = pattern_results
            self._write_generations_to_file(test_examples, pattern_results, method_name_str, pattern_name_str)
        
        null_pattern_name_str = "null_prompt"
        null_results = []
        logger.info(f"{method_name_str}: Generating with {null_pattern_name_str}")
        for example in test_examples:
            input_text = f"{example.premise} {example.hypothesis} {mask_token_str}"
            with torch.no_grad():
                inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                explanation = self._extract_explanation(generated_text, input_text)
                null_results.append(explanation)
        results[null_pattern_name_str] = null_results
        self._write_generations_to_file(test_examples, null_results, method_name_str, null_pattern_name_str)
        return results
    
    def _generate_synthetic_data(self, unlabeled_examples: List[ExplanationExample]) -> List[ExplanationExample]:
        logger.info(f"Generating synthetic data for {self.model_name}...")
        synthetic_examples = []
        self.model.eval()
        num_to_generate = min(len(unlabeled_examples), 1000) # As per original example
        mask_token_str = self._get_mask_token()
        
        for i, example in enumerate(unlabeled_examples[:num_to_generate]):
            if i % 100 == 0 and i > 0: logger.info(f"Synthetic data: {i}/{num_to_generate}")
            input_text = self.prompt_template.format_input(example, 0, 0, mask_token_str)
            with torch.no_grad():
                inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                synthetic_explanation = self._extract_explanation(generated_text, input_text)
                synthetic_examples.append(ExplanationExample(
                    premise=example.premise, hypothesis=example.hypothesis, label=example.label,
                    explanation=synthetic_explanation, guid=f"synthetic_{example.guid if example.guid else i}"
                ))
        logger.info(f"Finished generating {len(synthetic_examples)} synthetic examples for {self.model_name}.")
        return synthetic_examples

    def _fine_tune_model(self, dataset: ESNLIDataset, num_epochs: int, batch_size: int, learning_rate: float):
        logger.info(f"Fine-tuning {self.model_name} for {num_epochs} epochs on {len(dataset)} examples...")
        if len(dataset) == 0:
            logger.warning(f"Skipping fine-tuning for {self.model_name} as dataset is empty.")
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        optimizer = Adafactor(self.model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False)
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is None: # Should not happen with labels provided
                    logger.error("Loss is None during training. Skipping batch.")
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx > 0 and batch_idx % (max(1, len(dataloader) // 10)) == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
            logger.info(f"Epoch {epoch+1}/{num_epochs} for {self.model_name} completed. Avg loss: {avg_loss:.4f}")

    def method2_fine_tuning(self, train_examples: List[ExplanationExample],
                          unlabeled_examples: List[ExplanationExample],
                          test_examples: List[ExplanationExample],
                          num_epochs: int = 3, batch_size: int = 2, learning_rate: float = 1e-4) -> Dict[str, List[str]]:
        method_name_str = "method2_finetuning"
        logger.info(f"Running {method_name_str} for {self.model_name}")
        
        # This will use the *current* state of self.model to generate synthetic data.
        # If called on a fresh generator, self.model is M0.
        synthetic_examples = self._generate_synthetic_data(unlabeled_examples)
        
        combined_examples = train_examples + synthetic_examples
        if not combined_examples:
            logger.warning(f"{method_name_str}: No training data (labeled + synthetic). Skipping fine-tuning for {self.model_name}.")
            return {} # Return empty results if no data to train on
            
        combined_dataset = ESNLIDataset(combined_examples, self.tokenizer, self.prompt_template, is_training=True)
        self._fine_tune_model(combined_dataset, num_epochs, batch_size, learning_rate)
        
        evaluation_results = self._evaluate_fine_tuned_model(test_examples)
        for pattern_name, generations in evaluation_results.items():
            self._write_generations_to_file(test_examples, generations, method_name_str, pattern_name)
        return evaluation_results

    def _evaluate_fine_tuned_model(self, test_examples: List[ExplanationExample]) -> Dict[str, List[str]]:
        logger.info(f"Evaluating fine-tuned {self.model_name}...")
        self.model.eval()
        results = {}
        mask_token_str = self._get_mask_token()

        for pattern_id in range(4): # Test with each of the 4 main patterns
            pattern_name = f"pattern_{pattern_id}"
            pattern_results = []
            logger.info(f"Evaluating {self.model_name} with {pattern_name}...")
            for example in test_examples:
                input_text = self.prompt_template.format_input(example, pattern_id, 0, mask_token_str)
                with torch.no_grad():
                    inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    explanation = self._extract_explanation(generated_text, input_text)
                    pattern_results.append(explanation)
            results[pattern_name] = pattern_results
        
        null_pattern_name = "null_prompt"
        logger.info(f"Evaluating {self.model_name} with {null_pattern_name}...")
        null_results = []
        for example in test_examples:
            input_text = f"{example.premise} {example.hypothesis} {mask_token_str}"
            with torch.no_grad():
                inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                explanation = self._extract_explanation(generated_text, input_text)
                null_results.append(explanation)
        results[null_pattern_name] = null_results
        return results

    def _generate_ensemble_candidates(self, unlabeled_examples: List[ExplanationExample]) -> List[Tuple[ExplanationExample, List[str]]]:
        logger.info(f"Generating ensemble candidates (using M') for {self.model_name}...")
        candidate_data = []
        self.model.eval() # self.model is M' here
        num_to_process = len(unlabeled_examples)
        mask_token_str = self._get_mask_token()

        for i, example in enumerate(unlabeled_examples):
            if i % 100 == 0 and i > 0: logger.info(f"M' candidates for {self.model_name}: {i}/{num_to_process}")
            candidates = []
            for pattern_id in range(4):
                input_text = self.prompt_template.format_input(example, pattern_id, 0, mask_token_str)
                with torch.no_grad():
                    inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    explanation = self._extract_explanation(generated_text, input_text)
                    candidates.append(explanation)
            candidate_data.append((example, candidates))
        logger.info(f"Finished M' candidates for {num_to_process} unlabeled examples for {self.model_name}.")
        return candidate_data

    def _score_candidates(self, candidate_data: List[Tuple[ExplanationExample, List[str]]]) -> List[ExplanationExample]:
        logger.info(f"Scoring candidates (using M0) for {self.model_name}...")
        self.model.eval() # self.model is M0 here
        scored_examples_with_scores = []
        num_to_process = len(candidate_data)

        for i, (example, candidates) in enumerate(candidate_data):
            if i % 100 == 0 and i > 0: logger.info(f"M0 Scoring for {self.model_name}: {i}/{num_to_process}")
            best_score_for_this_example = float('-inf')
            best_explanation_for_this_example = candidates[0] if candidates else ""
            input_text_for_scoring = f"{example.premise} {example.hypothesis}"
            
            with torch.no_grad():
                inputs_for_scoring = self.tokenizer(input_text_for_scoring, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to(self.device)
                for candidate_explanation_text in candidates:
                    target_ids_dict = self.tokenizer(candidate_explanation_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
                    target_ids = target_ids_dict['input_ids'].to(self.device)
                    target_ids[target_ids == self.tokenizer.pad_token_id] = -100
                    outputs = self.model(input_ids=inputs_for_scoring['input_ids'], attention_mask=inputs_for_scoring['attention_mask'], labels=target_ids)
                    current_score = -outputs.loss.item() if outputs.loss is not None else float('-inf')
                    if current_score > best_score_for_this_example:
                        best_score_for_this_example = current_score
                        best_explanation_for_this_example = candidate_explanation_text
            
            scored_example_obj = ExplanationExample(
                premise=example.premise, hypothesis=example.hypothesis, label=example.label,
                explanation=best_explanation_for_this_example, guid=f"ensemble_scored_{example.guid if example.guid else i}"
            )
            scored_examples_with_scores.append((scored_example_obj, best_score_for_this_example))
        
        logger.info(f"Finished M0 scoring for {num_to_process} examples for {self.model_name}.")
        scored_examples_with_scores.sort(key=lambda item: item[1], reverse=True)
        cutoff_idx = int(0.8 * len(scored_examples_with_scores))
        final_scored_examples = [item[0] for item in scored_examples_with_scores[:cutoff_idx]]
        logger.info(f"Selected top {len(final_scored_examples)} scored examples after filtering for {self.model_name}.")
        return final_scored_examples

    def method3_ensembling(self, train_examples: List[ExplanationExample],
                         unlabeled_examples: List[ExplanationExample], test_examples: List[ExplanationExample],
                         num_epochs: int = 3, batch_size: int = 2, learning_rate: float = 1e-4) -> List[str]:
        method_name_str = "method3_ensembling"
        logger.info(f"Running {method_name_str} for {self.model_name}")

        original_model_name = self.model_name # To reload M0 correctly
        
        # Step 1: Fine-tune M0 on D_L + D_U_pseudo_M0 to get M'
        logger.info(f"Method 3, Step 1 (M'): Fine-tuning {self.model_name} (as M0) on labeled + M0-synthetic data.")
        # self.model is M0 at this point if it's a fresh generator, or after _load_model_and_tokenizer
        synthetic_examples_m0 = self._generate_synthetic_data(unlabeled_examples) # Uses current self.model (M0)
        combined_train_for_m_prime = train_examples + synthetic_examples_m0
        if not combined_train_for_m_prime:
            logger.warning(f"{method_name_str} Step 1: No data for M' training. Skipping fine-tuning for {self.model_name}.")
        else:
            m_prime_dataset = ESNLIDataset(combined_train_for_m_prime, self.tokenizer, self.prompt_template, is_training=True)
            self._fine_tune_model(m_prime_dataset, num_epochs, batch_size, learning_rate) # self.model becomes M'

        # Step 2: Generate candidates for D_U using M'
        logger.info(f"Method 3, Step 2 (M'): Generating candidates using fine-tuned M' ({self.model_name}).")
        candidate_data_from_m_prime = self._generate_ensemble_candidates(unlabeled_examples) # Uses M'

        # Step 3: Score candidates with M0 (untrained model)
        logger.info(f"Method 3, Step 3 (M0): Reloading base {original_model_name} as M0 for scoring.")
        self._load_model_and_tokenizer() # Reloads M0 for self.model_name
        scored_data = self._score_candidates(candidate_data_from_m_prime) # Uses self.model (now M0)

        # Step 4: Train M_E from M0 on D_L U D_U_scored
        logger.info(f"Method 3, Step 4 (M_E): Fine-tuning {self.model_name} (as M0) on labeled + scored data for M_E.")
        final_ensemble_train_examples = train_examples + scored_data
        if not final_ensemble_train_examples:
            logger.warning(f"{method_name_str} Step 4: No data for M_E training. Skipping fine-tuning for {self.model_name}.")
            final_predictions = ["NO M_E TRAINING DATA"] * len(test_examples) if test_examples else []
        else:
            ensemble_dataset = ESNLIDataset(
                final_ensemble_train_examples, self.tokenizer, self.prompt_template,
                pattern_ids=[0], prefix_ids=[0], is_training=True # Only null pattern P0
            )
            self._fine_tune_model(ensemble_dataset, num_epochs, batch_size, learning_rate) # self.model becomes M_E
            
            # Step 5: Generate final predictions with M_E
            logger.info(f"Method 3, Step 5 (M_E): Generating final predictions with M_E ({self.model_name}).")
            final_predictions = self._generate_ensemble_predictions(test_examples) # Uses M_E
        
        self._write_generations_to_file(test_examples, final_predictions, method_name_str, "final_ensemble_predictions")
        return final_predictions

    def _generate_ensemble_predictions(self, test_examples: List[ExplanationExample]) -> List[str]:
        logger.info(f"Generating ensemble predictions (using M_E) for {self.model_name}...")
        self.model.eval() # self.model is M_E
        predictions = []
        mask_token_str = self._get_mask_token()
        for example in test_examples:
            input_text = f"{example.premise} {example.hypothesis} {mask_token_str}" # Null prompt
            with torch.no_grad():
                inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                explanation = self._extract_explanation(generated_text, input_text)
                predictions.append(explanation)
        return predictions

def load_esnli_data_from_hf(split: str, num_samples: Optional[int] = None) -> List[ExplanationExample]:
    logger.info(f"Loading e-SNLI '{split}' split from Hugging Face datasets (max {num_samples} samples)...")
    try:
        # Load the full split first, then select if num_samples is specified.
        # This is more robust if the dataset doesn't directly support num_samples in load_dataset.
        dataset_full = load_dataset("esnli", split=split)
        if num_samples is not None and num_samples < len(dataset_full):
             dataset = dataset_full.select(range(num_samples))
        else:
            dataset = dataset_full
    except Exception as e:
        logger.error(f"Failed to load e-SNLI dataset '{split}': {e}", exc_info=True)
        return []

    label_map = {idx: name.upper() for idx, name in enumerate(dataset.features['label'].names)}
    examples = []
    
    for i, hf_example in enumerate(dataset):
        gold_explanation = hf_example.get('explanation_1', "")
        if isinstance(gold_explanation, list): gold_explanation = gold_explanation[0] if gold_explanation else ""
        
        examples.append(ExplanationExample(
            premise=hf_example.get('premise', ""), hypothesis=hf_example.get('hypothesis', ""),
            label=label_map.get(hf_example.get('label'), "NEUTRAL"), explanation=gold_explanation,
            guid=hf_example.get('pairID', f"{split}_{i}")
        ))
    logger.info(f"Loaded {len(examples)} examples from e-SNLI '{split}' split.")
    return examples

# Example usage
# Example usage
if __name__ == "__main__":
    samples = [500] #[10, 100, 500]
    base_output_directory = "generations" # Or your preferred name
    os.makedirs(base_output_directory, exist_ok=True)

    for num_samples in samples:
        # Load data once, outside the model loop
        logger.info("Loading dataset splits...")
        # Using very small numbers for quick testing. Increase for actual runs.
        train_examples_subset = load_esnli_data_from_hf(split="train", num_samples=num_samples) # Reduced for faster test
        test_examples_subset = load_esnli_data_from_hf(split="test", num_samples=num_samples//2)   # Reduced for faster test
        unlabeled_examples_subset = load_esnli_data_from_hf(split="validation", num_samples=num_samples//2) # Reduced

        if not train_examples_subset or not test_examples_subset:
            logger.error("Not enough data to run example. Exiting.")
            exit()

        models_to_run_config = [
            {"name": "facebook/bart-base", "short_name": "bart-base"}, # Added BART back for completeness
            #{"name": "google-t5/t5-small", "short_name": "t5-small"},      # Changed to t5-small for faster testing
            #{"name": "google/pegasus-cnn_dailymail", "short_name": "pegasus-cnn_dailymail"}, # Smaller pegasus
            {"name": "google-t5/t5-base", "short_name": "t5-base"}, # Uncomment to use t5-base
            {"name": "google/pegasus-xsum", "short_name": "pegasus-xsum"}, # Uncomment for XSUM variant
        ]

        for model_config in models_to_run_config: # Iterate through the defined model configurations
            model_name = model_config["name"]
            model_short_name = model_config["short_name"]
            
            logger.info(f"\n\n{'='*25} RUNNING EXPERIMENTS FOR MODEL: {model_name} ({model_short_name}) {'='*25}")
            
            model_specific_output_dir = os.path.join(base_output_directory, f"{num_samples}", model_short_name)
            # os.makedirs(model_specific_output_dir, exist_ok=True) # This is already done in ExplanationGenerator constructor

            try:
                # Instantiate generator for the current model
                generator = ExplanationGenerator(model_name=model_name, output_dir=model_specific_output_dir)
                
                # --- Run Method 1: Unsupervised ---
                logger.info(f"\n--- Method 1: UNSUPERVISED for {model_name} ---")
                unsupervised_results = generator.method1_unsupervised(test_examples_subset)
                if unsupervised_results and test_examples_subset and unsupervised_results.get("null_prompt"): # Check if list not empty
                    logger.info(f"M1 results for {model_name} (first ex, null_prompt): {unsupervised_results['null_prompt'][0]}")

                # --- Run Method 2: Fine-tuning ---
                # Re-initialize generator to ensure it starts from the base pre-trained model for this method's specific fine-tuning
                generator_m2 = ExplanationGenerator(model_name=model_name, output_dir=model_specific_output_dir)
                logger.info(f"\n--- Method 2: FINE-TUNING for {model_name} ---")
                if unlabeled_examples_subset:
                    finetuning_results = generator_m2.method2_fine_tuning(
                        train_examples_subset, unlabeled_examples_subset, test_examples_subset,
                        num_epochs=1, batch_size=1 # Even smaller batch for large models if memory is tight
                    )
                    if finetuning_results and test_examples_subset and finetuning_results.get("null_prompt"): # Check if list not empty
                        logger.info(f"M2 results for {model_name} (first ex, null_prompt): {finetuning_results['null_prompt'][0]}")
                else:
                    logger.warning(f"Skipping Method 2 for {model_name} due to lack of unlabeled data.")

                # --- Run Method 3: Ensembling ---
                # Re-initialize for method 3 to ensure M0 is fresh.
                generator_m3 = ExplanationGenerator(model_name=model_name, output_dir=model_specific_output_dir)
                logger.info(f"\n--- Method 3: ENSEMBLING for {model_name} ---")
                if unlabeled_examples_subset and len(unlabeled_examples_subset) >= 5:
                    ensemble_results = generator_m3.method3_ensembling(
                        train_examples_subset, unlabeled_examples_subset, test_examples_subset,
                        num_epochs=1, batch_size=1 # Even smaller batch
                    )
                    if ensemble_results and test_examples_subset: # Check if list not empty
                        logger.info(f"M3 results for {model_name} (first ex): {ensemble_results[0]}")
                else:
                    logger.warning(f"Skipping Method 3 for {model_name} due to insufficient unlabeled data for ensembling (need at least 5).")

            except Exception as e:
                logger.error(f"Error during experiments for model {model_name}: {e}", exc_info=True)
                logger.error(f"Skipping remaining methods for {model_name} and moving to next model if any.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # Clear GPU memory if on CUDA after each model successfully processes
            if torch.cuda.is_available():
                # Ensure generators are deleted if they were created
                if 'generator' in locals(): del generator
                if 'generator_m2' in locals(): del generator_m2
                if 'generator_m3' in locals(): del generator_m3
                torch.cuda.empty_cache()
                logger.info(f"Cleared CUDA cache after processing model {model_name}")

        print(f"\nAll model experiment runs completed. Check the '{base_output_directory}' directory for saved generations.")