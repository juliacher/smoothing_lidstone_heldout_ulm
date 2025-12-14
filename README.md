# Smoothing Techniques for Unigram Language Models  
**A Comparison of Lidstone and Held-Out Estimation**

## Overview
This repository contains the full source code and LaTeX materials accompanying the paper  
*“Smoothing Techniques for Unigram Language Models: A Comparison of Lidstone and Held-Out Estimation”*.

The project investigates classical smoothing methods for unigram language models, focusing on how probability mass is redistributed to rare and unseen words. We implement and compare **Lidstone smoothing** and **held-out estimation**, using **perplexity** as the primary evaluation metric.

All experiments are conducted on derived subsets of the **Reuters-21578 corpus**, with careful separation of training, validation, and test data.

---

## Methods
- **Baseline model:** Unigram language model with Maximum Likelihood Estimation (MLE)
- **Smoothing techniques:**
  - Lidstone smoothing (with parameter tuning on validation data)
  - Held-out estimation (using counts-of-counts)
- **Evaluation metric:** Perplexity
- **Vocabulary size:** Fixed upper bound of |V| = 300,000

The implementation includes parameter sweeps for Lidstone smoothing, construction of held-out frequency classes, and verification that all probability distributions are properly normalized.

---

## Results
- Unsmoothed MLE assigns zero probability to unseen words and yields infinite perplexity.
- Lidstone smoothing with a small, tuned parameter achieves the **lowest test perplexity**.
- Held-out estimation provides a more flexible redistribution of probability mass to rare and unseen events, though with slightly higher perplexity.
- Results highlight the pedagogical and conceptual value of classical smoothing methods, even in the context of modern neural language models.

---

## Repository Structure
- `src/` – Python source code for data processing, model implementation, and evaluation  
- `figures/` – Generated plots and figures used in the paper  
- `data/` – Scripts or instructions for preparing the Reuters-21578 subsets  
- `latex/` – LaTeX source files for the paper  
- `README.md` – Project overview (this file)

---

## Tools and Technologies
- Python (NumPy, Pandas)
- Matplotlib (visualization)
- LaTeX (document preparation)

AI-supported tools were used **only for spelling and grammar correction of the text**, not for scientific content, analysis, or results.

---

## Author
Julia Cher  
University of Basel  
December 2025

---

## License
This repository is provided for academic and educational purposes.
