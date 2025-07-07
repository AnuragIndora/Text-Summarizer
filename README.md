# 🧠 Transformer-Based Text Summarizer

A scalable, modular, and memory-efficient **text summarization system** using `facebook/bart-base`, built with full MLOps-style pipelines for data ingestion, validation, transformation, training, evaluation, and deployment.

> ⚙️ Designed for research and real-world deployment with CLI, Streamlit frontend, and FastAPI backend support.

---

## 🚀 Project Highlights

* **Model**: `facebook/bart-base` via Hugging Face
* **Dataset**: `cnn_dailymail` v3.0.0 (`abisee/cnn_dailymail`)
* **Training Data Used**: 50K samples (scalable to full 287K)
* **Validation**: 3K (notebook) / 1K (pipeline) (out of 13.4K)
* **Test**: 3K (notebook) / 1K (pipeline) (out of 11.5k)
* **Evaluation**: ROUGE-L: `20.67`, ROUGE-1: `25.21`, ROUGE-2: `12.28`
* **Frameworks**: PyTorch, Hugging Face Transformers + Accelerate
* **Deployment**: Dockerized, Streamlit UI, FastAPI backend

---

## 🔁 Pipeline Flow

> Implemented using modular stages, YAML-config driven execution

1. **Data Ingestion** → `cnn_dailymail` load and split
2. **Data Validation** → schema check, format verification
3. **Data Transformation** → tokenization and formatting
4. **Model Training** → with `Seq2SeqTrainer`
5. **Model Evaluation** → on custom-held-out test set
6. **Prediction** → CLI or UI-based summarization

---

## ⚡ Accelerated Training (for low VRAM GPUs)

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,                      # Mixed-precision
    num_train_epochs=3,
    predict_with_generate=True
)
```

Trained on **4GB RTX 3050** using Hugging Face Accelerate for speed and efficiency.

---

## 📊 Evaluation Results

```
Train Loss: 0.9688 | Global Steps: 18,750
Test Loss: 0.9473 | ROUGE-1: 25.21 | ROUGE-2: 12.28 | ROUGE-L: 20.67
```

---

## 🖥 UI + API Support

### 🔹 CLI

Run training or prediction:

```bash
python main.py
```

### 🔹 Streamlit Frontend

```bash
streamlit run frontend_app.py
```

### 🔹 FastAPI Backend

```bash
uvicorn backend_app:app --reload
```

---

## 🐳 Docker Support

Build and run container:

```bash
docker build -t summarizer-app .
docker run -p 8000:8000 summarizer-app
```

---

## 🤝 Contribution

This is a solo project but **open to contributors**.
Feel free to raise issues, suggest features, or submit PRs 🚀

---

## 📜 License

MIT License

---

## 📌 TODO (Future Work)

* Add support for `peft` / LoRA fine-tuning
* Implement memory-efficient attention (FlashAttention)
* Deploy to Hugging Face Spaces or Streamlit Cloud
* Add multi-lingual summarization support
