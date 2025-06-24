
## Prerequisites

1.  **Python Environment:** Python 3.9+ is recommended.
2.  **Conda (Recommended):** For managing dependencies and environments.
3.  **GPU (Recommended):** For faster training and inference, especially with larger models. Ensure CUDA drivers and toolkit are installed if using an NVIDIA GPU.

## Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate Conda Environment:**
    It's highly recommended to create a dedicated Conda environment to avoid conflicts.
    ```bash
    conda create -n expgen python=3.12 -y  # Or your preferred Python version
    conda activate expgen
    ```

3.  **Install PyTorch:**
    Install PyTorch according to your system and CUDA version. Visit [pytorch.org](https://pytorch.org/) for the correct command.
    Example for CUDA 11.8:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```
    For CPU-only:
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```

4.  **Install Other Dependencies:**
    The scripts require several Python libraries. `sentencepiece` is crucial for T5 and Pegasus tokenizers.
    ```bash
    conda install -c conda-forge transformers datasets sentencepiece protobuf absl-py -y
    conda install -c conda-forge scikit-learn nltk rouge_score -y
    # Or using pip after activating the environment:
    # pip install transformers datasets sentencepiece protobuf absl-py scikit-learn nltk rouge_score
    ```
    *   `transformers`: For Hugging Face models and tokenizers.
    *   `datasets`: For loading the e-SNLI dataset from Hugging Face Hub.
    *   `sentencepiece`: Tokenizer library required by T5 and Pegasus.
    *   `protobuf`, `absl-py`: Dependencies for `sentencepiece`.
    *   `scikit-learn`: (Currently not directly used by the provided core scripts but often useful).
    *   `nltk`: For BLEU score calculation.
    *   `rouge_score`: For ROUGE score calculation.

5.  **Download NLTK Data (for BLEU score):**
    Run this in a Python interpreter within your activated Conda environment:
    ```python
    import nltk
    nltk.download('punkt') # Required by sentence_bleu if not already present
    ```

## Running the Experiments (`gmn_new.py`)

The `gmn_new.py` script handles data loading, model fine-tuning, and explanation generation for the specified models and methods.

1.  **Configure Models (Optional):**
    Open `gmn_new.py`. At the bottom, in the `if __name__ == "__main__":` block, you can configure which models to run:
    ```python
    models_to_run_config = [
        {"name": "facebook/bart-base", "short_name": "bart-base"},
        {"name": "google-t5/t5-base", "short_name": "t5-base"},
        {"name": "google/pegasus-xsum", "short_name": "pegasus-xsum"},
        # You can add/remove models or use smaller versions for faster testing:
        # {"name": "t5-small", "short_name": "t5-small"},
        # {"name": "google/pegasus-cnn_dailymail", "short_name": "pegasus-cnn_dailymail"},
    ]
    ```
    The script also uses small subsets of data (`num_samples`) and few epochs/small batch sizes for quick testing. Increase these for full experimental runs.

2.  **Set Output Directory (Optional):**
    The default base output directory is defined as:
    ```python
    base_output_directory = "generations"
    ```
    Model-specific results will be saved in subdirectories under this path (e.g., `generations/bart-base/`).

3.  **Execute the Script:**
    Run the script from your terminal within the activated Conda environment:
    ```bash
    python gmn_new.py
    ```
    This process can take a significant amount of time, especially with larger models or more data/epochs. Monitor the console output for progress and potential errors.

    The script will:
    *   Load the e-SNLI dataset from Hugging Face Hub.
    *   For each model in `models_to_run_config`:
        *   Instantiate the `ExplanationGenerator`.
        *   Run Method 1 (Unsupervised generation).
        *   Run Method 2 (Fine-tuning with synthetic data).
        *   Run Method 3 (Ensembling with knowledge distillation).
        *   Save generated explanations in `.jsonl` files under `generations/<model_short_name>/<method_name>/`.

## Evaluating the Generations (`eval.py`)

The `eval.py` script calculates BLEU-1, ROUGE-1 F1, and BERT-Score F1 for the generated explanations.

1.  **Ensure Generations Exist:**
    Make sure `gmn_new.py` has completed successfully and generated the output files in the `generations/` directory (or your configured output directory).

2.  **Configure Models for Evaluation (Crucial):**
    Open `eval.py`. In the `if __name__ == "__main__":` block, ensure the `model_short_names` list matches the `short_name`s of the models for which you ran experiments and want to evaluate:
    ```python
    # This should match 'base_output_directory' from your experiment script
    results_base_dir = "generations"

    # THIS LIST MUST MATCH THE 'short_name's FROM YOUR EXPERIMENT SCRIPT
    model_short_names = ["bart-base", "t5-base", "pegasus-xsum"]
    ```
    Also, ensure `results_base_dir` matches the output directory used by `gmn_new.py`.

3.  **Execute the Evaluation Script:**
    Run the script from your terminal:
    ```bash
    python eval.py
    ```
    The script will:
    *   Iterate through each model specified in `model_short_names`.
    *   For each model, it will look for generation files in the expected subdirectories (e.g., `generations/bart-base/method1_unsupervised/null_prompt_generations.jsonl`).
    *   Calculate BLEU-1, ROUGE-1 F1, and BERT-Score F1.
    *   Print the scores to the console.
    *   Save a comprehensive summary of all scores to a CSV file named `evaluation_summary_all_models.csv` inside the `results_base_dir`.

## Output Structure

*   **`generations/`**:
    *   **`<model_short_name>/`** (e.g., `bart-base`, `t5-base`)
        *   **`method1_unsupervised/`**:
            *   `null_prompt_generations.jsonl`
            *   `pattern_0_generations.jsonl`
            *   ... (files for other patterns)
        *   **`method2_finetuning/`**:
            *   `null_prompt_generations.jsonl`
            *   `pattern_0_generations.jsonl`
            *   ...
        *   **`method3_ensembling/`**:
            *   `final_ensemble_predictions_generations.jsonl`
    *   **`evaluation_summary_all_models.csv`**: CSV file containing all evaluation scores.

Each `.jsonl` generation file contains one JSON object per line, with fields like: `guid`, `premise`, `hypothesis`, `gold_label`, `gold_explanation`, and `generated_explanation`.

## Troubleshooting

*   **`ImportError: ... sentencepiece ...`**: Ensure `sentencepiece` is installed correctly in your active Conda environment (`pip install sentencepiece` or `conda install -c conda-forge sentencepiece`).
*   **`ImportError: ... libprotobuf-lite.so ... undefined symbol ...`**: This usually indicates a C++ library version mismatch, often with `protobuf` and `absl-py`. The most robust solution is to create a fresh Conda environment and install dependencies carefully, preferably using `conda-forge` for complex packages. (See Step 3 in Setup Instructions).
*   **CUDA/GPU Issues (`OutOfMemoryError`)**:
    *   Reduce `batch_size` in `gmn_new.py` (e.g., to 1 or 2).
    *   Use smaller model variants (e.g., `t5-small` instead of `t5-base`).
    *   Ensure no other processes are consuming significant GPU memory.
    *   The script includes `torch.cuda.empty_cache()` calls, which can help but may not always be sufficient for very large models on limited hardware.
*   **File Not Found in `eval.py`**:
    *   Verify that `results_base_dir` in `eval.py` matches the output directory of `gmn_new.py`.
    *   Check that `model_short_names` in `eval.py` correctly lists the models you ran.
    *   Confirm that the `method_configs` in `eval.py` reflect the actual filenames and subdirectory names produced by `gmn_new.py`.