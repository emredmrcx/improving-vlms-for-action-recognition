# This repository contains source codes for the term project of YZV302E Deep Learning course. 
- Checkpoints: [link](https://drive.google.com/drive/folders/1KxsQNCtPpLqCeCX5NZxxOp4Q5jLmTKDg?usp=sharing)  
- Sampled Frames & Evaluation Results: [link](https://drive.google.com/drive/folders/1qmaKnEHplI70TnOekgJdArW4r5rW5_EJ?usp=sharing)


# Improving VLMs for Action Recognition

This repository contains Jupyter notebooks focused on improving **Video–Language Models (VLMs)** for action recognition tasks, specifically using the **EPIC-KITCHENS 55 (EK55)** dataset.  
The project reformulates action recognition as a **multiple-choice question (MCQ)** task by introducing hard, visually grounded distractors, followed by fine-tuning and evaluation of state-of-the-art VLMs, specifically **Qwen3-VL**.

The core goal is to evaluate whether modern VLMs genuinely rely on visual evidence when confronted with semantically plausible but visually inconsistent answer options, rather than exploiting language-only shortcuts.

---

## Project Structure

The notebooks are organized into three main functional stages that mirror the experimental pipeline:

### 1. Distractor Generation (`notebooks/distractor_generation`)
This stage focuses on generating challenging negative answer options to construct robust MCQ benchmarks.

- **`ek55_avion_distractors_with_filter.ipynb`**  
  Generates high-quality distractors using the pre-trained **AVION** video–text model. A filtering strategy based on top-k predictions is applied to enforce semantic plausibility and diversity while maintaining visual inconsistency with the ground-truth action.

- **`ek55_avion_distractors_without_filter.ipynb`**  
  A baseline AVION-based distractor generation pipeline without semantic filtering, used to analyze the effect of raw embedding similarity.

- **`EgoVLPv2_Distractor_Generation.ipynb`**  
  Uses the **EgoVLPv2** model to generate temporally-aware hard negatives. This notebook includes environment setup for Python 3.10 and compatibility patches required for the EgoVLPv2 codebase.

- **`CLIP.ipynb`**  
  Evaluates zero-shot action recognition using **CLIP**. Video representations are obtained by aggregating frame-level embeddings, which are compared against text-encoded action descriptions to compute top-k rankings.

- **`TSM_TRN_MTRN_TSN.ipynb`**  
  Implements and evaluates classical video-only action recognition models (TSM, TRN, MTRN, TSN), providing a non-VLM baseline for comparison.

---

### 2. Fine-Tuning (`notebooks/finetuning`)
This stage performs parameter-efficient fine-tuning (PEFT) of large VLMs using the **Unsloth** framework.

- **`Qwen3_VL_Finetuning_with_vision.ipynb`**  
  Jointly fine-tunes the **Qwen3-VL-2B** model, updating the vision encoder, language model, attention layers, and MLP modules. This setup enables learning aligned visual–linguistic representations under hard distractor supervision.

- **`Qwen3_VL_Finetuning_without_vision.ipynb`**  
  Fine-tunes only the language, attention, and MLP components while keeping the vision encoder frozen. This configuration serves as a language-heavy baseline to quantify the contribution of visual learning.

---

### 3. Inference (`notebooks/inference`)
This stage handles large-scale evaluation and analysis.

- **`ek55_mcq_inference.ipynb`**  
  The primary inference and evaluation notebook. It benchmarks multiple generations of **Qwen-VL** models (including Qwen2, Qwen2.5, and Qwen3) on EK55 MCQ tasks, reporting accuracy metrics and detailed prediction logs across different distractor settings.

---

## Technical Requirements

### Dataset
- **EPIC-KITCHENS 55 (EK55)** sampled video frames and action annotations.

### Key Libraries and Frameworks
- **PyTorch** – core deep learning framework  
- **HuggingFace Transformers** – model loading and tokenization  
- **Unsloth / FastVisionModel** – parameter-efficient fine-tuning of VLMs  
- **Decord** – efficient video frame loading  
- **NumPy / Pandas** – data processing and analysis  
- **Scikit-learn** – evaluation utilities and metrics  
- **OpenCLIP / CLIP** – vision–language embedding baselines  
- **AVION** – video–text embedding model  
- **EgoVLPv2** – temporally-aware video–language model  

### Platform
- Optimized for **Google Colab**, with **Google Drive** integration for dataset access, checkpoint storage, and experiment logging.

---

## Notes

- Notebooks are designed to be runnable independently but follow the intended pipeline:  
  **distractor generation → fine-tuning → inference**.
- Dataset paths, checkpoints, and environment variables may require adjustment depending on the execution environment.
