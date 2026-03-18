# DeepCausalPV: Adverse Drug Effect Detection from FAERS Data

> **Note:** This repository is based on the codes and data from [https://github.com/XingqiaoWang/DeepCausalPV-master](https://github.com/XingqiaoWang/DeepCausalPV-master).

Welcome to the **DeepCausalPV** source code repository. This project aims to detect and analyze adverse drug effects using data from the FDA Adverse Event Reporting System (FAERS). By leveraging state-of-the-art transformer models (ALBERT, BioBERT) alongside Large Language Models (LLaMA, DeepSeek API), we identify critical causal effects associated with clinical terms (e.g., age, dose, indication, secondary suspect drugs).

---

## 📂 Repository Structure

The `src/` directory contains four main folders, categorized by their specific purpose in the pipeline:

### 1. `Analgesics-induced_acute_liver_failure/`
This folder contains the complete pipeline for analyzing acute liver failure induced by analgesics.
- **Transformer Training:** Scripts to fine-tune `ALBERT` and `BioBERT` on the clinical data (`albert_train.py`, `biobert_train.py`).
- **LLM Integration:** Converts tabular features into cohesive sentences using the DeepSeek API (`create_llm_sent.py`) and trains models specifically on these sentences (`biobert_llm_train.py`).
- **Evaluation & Statistics:** Notebooks to compute Z-scores from model predictions and calculate feature significance with T-tests (`calc_zscores.ipynb`, `student_test.ipynb`).

### 2. `Tramadol-related_mortalities/`
This folder contains the exact same pipeline structure as the Analgesics folder but is specifically adapted for analyzing factors leading to Tramadol-related mortalities. The scripts and notebooks operate identically to the ones described above.

### 3. `extra_training_codes_for_llama/`
Dedicated scripts for fine-tuning the `Med-LLaMA3-8B` model using LoRA quantization.
- **`install.sh`**: Prepares the Conda environment and installs GPU dependencies.
- **`run_finetune.sh`**: Submits the SLURM job to start the LLaMA fine-tuning process.
- **`llama_train.py`**: The core PyTorch/HuggingFace script handling the dataset mapping, LoRA configuration, and trainer logic.

### 4. `extra_evaluation_codes/`
Advanced post-training evaluation notebooks.
- **`calc_metrics.ipynb`**: Aggregates predictions across XGBoost, ALBERT, BioBERT, and LLaMA, calculating comprehensive metrics including AUC, Precision, Recall, F1, Accuracy, and Expected Calibration Error (ECE).
- **`vis_trees.ipynb`**: Extrapolates significant clinical terms using calculated Z-scores and generates hierarchical pyramid visualizations (saved as PDFs) to interpret the factors driving adverse effect predictions.

---

## 🚀 How to Run the Pipeline

### Prerequisites
- Python 3.9+
- A valid `dat/` folder containing your processed datasets.
- CUDA-compatible GPU (Highly recommended for transformer training).
- Primary dependencies: `torch`, `transformers`, `pandas`, `scikit-learn`, `networkx`, `matplotlib`, `openai`, `datasets`, `peft` and `bitsandbytes`.

### Step-by-Step Execution

**Step 1: Data Setup**
Ensure your working directory has the following structure so the dynamic relative paths work correctly across all scripts:
```text
DeepCausalPV-project/
├── dat/              # Processed CSV files and cross-validation splits
└── src/              # This directory
```

**Step 2: Generate Sentences (Optional)**
If you wish to utilize LLM-generated sentences from tabular data, navigate into the desired condition folder (e.g., `Analgesics-induced_acute_liver_failure`) and run:
```bash
python create_llm_sent.py
```


**Step 3: Train Standard Transformers**
To train ALBERT or BioBERT loops across all cross-validation splits, navigate into the condition folder and run:
```bash
python albert_train.py
python biobert_train.py
```

**Step 4: Train LLaMA (HPC Clusters)**
Navigate to `extra_training_codes_for_llama` and use the SLURM dispatcher if on an HPC node:
```bash
sbatch install.sh
sbatch run_finetune.sh
```

**Step 5: Visualizations & Final Metrics**
After model training outputs are saved dynamically into `../../dat/<dataset>/proc/`, open and execute the notebooks in `extra_evaluation_codes/notebooks/` to calculate the final scalar performance metrics (`calc_metrics.ipynb`) and generate the visual decision trees representing causal feature importance (`vis_trees.ipynb`).
