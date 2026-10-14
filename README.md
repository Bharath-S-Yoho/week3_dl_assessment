# Week 3 Assessment — Beginner Version

This project contains the four tasks from the Week 3 Deep Learning + NLP assessment.

## Folder structure
- `Task1_Neural_Network/` — NumPy XOR network, functions, autograd, loss plots.
- `Task2_PyTorch/` — breast-cancer classification with PyTorch and scikit-learn.
- `Task3_NLP_Baseline/` — 300+ short labelled texts, TF-IDF baseline, embeddings.
- `Task4_Transformers_API/` — DistilBERT tokenizer/pipelines, fine-tuning script, FastAPI app.

## 1. Create and activate a virtual environment (optional but recommended)
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install packages for the task you want to run. The requirements files are kept separate because Transformer packages are heavier.

## 2. Run the tasks

### Task 1
```bash
pip install numpy matplotlib torch
cd Task1_Neural_Network
python task1.py
```
It saves `loss_curve.png` and `learning_rates.png`.

### Task 2
```bash
pip install numpy pandas scikit-learn torch matplotlib
cd Task2_PyTorch
python task2.py
```
The breast-cancer dataset is included in scikit-learn, so no separate CSV download is needed. The class is named `CsvDataset` as requested and implements `__len__` and `__getitem__`.

### Task 3
```bash
pip install pandas scikit-learn matplotlib gensim
cd Task3_NLP_Baseline
python task3.py
```
The script creates a labelled dataset of 360 short example texts locally. It tries to load GloVe vectors through Gensim. The first run may download vectors and requires internet. If GloVe is unavailable, the script reports the issue and still completes the TF-IDF work. The sentence-vector comparison is run when vectors load successfully.

### Task 4
```bash
pip install -r Task4_Transformers_API/requirements.txt
cd Task4_Transformers_API
python inspect_pipelines.py
```
This downloads pretrained models the first time and needs internet. It prints tokenizer outputs for two sentences and runs sentiment-analysis and NER on three sentences each.

Fine-tuning (2 epochs; use GPU/Colab if available):
```bash
python finetune.py
```
This uses a small subset of the labelled example data so it is practical for a beginner. It saves the model to `saved_model/` and reports accuracy and macro F1.

Start the API after the fine-tuned model is saved:
```bash
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs`, choose `POST /predict`, enter `{"text":"I love this product"}`, and click Execute. Empty text returns HTTP 422 due to request validation.

**Important:** The API expects the fine-tuned model in `saved_model/`. Run `finetune.py` first. If the model download or fine-tuning cannot run on your computer, use Google Colab with a GPU. Do not report example numbers as your actual measured results.

## 3. Git upload steps
Open a terminal in the main `Week3_Assessment_Beginner` folder.

If your existing GitHub repository is already cloned, copy these task folders into that repository instead, then run:
```bash
git status
git add .
git commit -m "Complete Week 3 Deep Learning and NLP assessment"
git push
```

If you have not cloned the existing repository yet:
```bash
git clone YOUR_EXISTING_REPOSITORY_URL
```
Copy the four task folders and this README into the cloned repository folder, then run the three Git commands above from inside that folder.

Replace `YOUR_EXISTING_REPOSITORY_URL` with your real repository URL. Do not upload passwords, access tokens, virtual environments, or very large downloaded model files. The `.gitignore` in Task 4 excludes the saved model folder; if your trainer requires model weights to be submitted, ask whether to upload them or share a model link because they can be large.
