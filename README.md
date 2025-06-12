# BillSum Text-Summarization Project

A lightweight sequence-to-sequence pipeline that turns lengthy U.S. Congressional and California-state bills into concise, readable summaries.

---

## Dataset
| Name | Rows | Description |
|------|------|-------------|
| **BillSum** | ≈23 K bills | Official text + expert summary for U.S. Congress (2013-2018) and CA state bills (2015-2016). Sourced from Hugging Face Datasets. (https://huggingface.co/datasets/FiscalNote/billsum) |

---

## About the Project

1. **Pre-processing**  
   Tokenisation, truncation/padding and label alignment handled via *transformers* `AutoTokenizer`.  
2. **Models Implemented**  
   - **RNN Seq2Seq** (SimpleRNN encoder/decoder)  
   - **LSTM Seq2Seq**  
   - **GRU Seq2Seq** *(best; ROUGE-1/2/L ≈ 0.95/0.92/0.95)* :contentReference[oaicite:1]{index=1}  
3. **Training & Tracking**  
   - 2 epochs per model using *TensorFlow/Keras*  
   - Experiment metrics logged with **Neptune.ai**  
4. **Evaluation**  
   Custom inference loop + Hugging Face `rouge` for ROUGE-1/2/L scoring.

---

## Key Features
- **Plug-and-play pipeline** – swap in any encoder-decoder model with minimal code changes.  
- **Clean metrics script** – single command prints ROUGE for dev/test splits.  
- **Reproducible notebooks** – all steps (tokenise -> train -> evaluate) captured in Jupyter.  
- **Neptune.ai dashboard** – live loss/ROUGE curves for every run.  
- **Modular code** – preprocessing, model definition and evaluation isolated in their own modules.

---

## Quick Start

```bash
git clone https://github.com/<your-org>/billsum-summarizer.git
cd billsum-summarizer
pip install -r requirements.txt

# train GRU model
python train.py --model gru --epochs 2

# evaluate saved checkpoint
python evaluate.py --checkpoint checkpoints/gru_best.h5
