📘 Embedding Assignment - NLP Project
📌 Objective
This assignment explores different word embedding techniques for Natural Language Processing tasks. The goal is to transform raw text into meaningful numerical representations using:

TF-IDF
Word2Vec
GloVe
FastText

Each method is implemented, compared, and optionally used in downstream tasks like classification or clustering.

🧰 Requirements
Install dependencies:

bash
Copy
Edit
pip install numpy pandas scikit-learn gensim tensorflow tqdm
📂 Project Structure
graphql
Copy
Edit
embedding-assignment/
├── data/
│   └── corpus.txt                   # Text corpus for training embeddings
├── embeddings/
│   ├── glove.6B.100d.txt            # Pre-trained GloVe vectors (downloaded)
│   └── custom_vectors.txt           # Trained Word2Vec or FastText vectors (optional)
├── notebook.ipynb                   # Main analysis notebook
├── glove_utils.py                   # GloVe loading utility
├── README.md                        # This file

🔍 Methods Implemented
1️⃣ TF-IDF (Term Frequency-Inverse Document Frequency)
Vectorizes text using frequency-based weighting

Uses sklearn.feature_extraction.text.TfidfVectorizer

2️⃣ Word2Vec
Trains word embeddings using the Skip-gram or CBOW model

Implemented using gensim.models.Word2Vec

3️⃣ GloVe
Uses pre-trained vectors (glove.6B.100d.txt)

Mapped via a tokenizer's word index to form an embedding matrix

Includes optional steps for custom GloVe training

4️⃣ FastText
Embeddings from subwords using gensim.models.FastText

Useful for out-of-vocabulary (OOV) words

📥 How to Use Pre-trained GloVe
Download GloVe vectors:
https://nlp.stanford.edu/projects/glove/

Place glove.6B.100d.txt into the embeddings/ folder.

Load using:

python
Copy
Edit
from glove_utils import load_glove_embeddings
glove_dict = load_glove_embeddings("embeddings/glove.6B.100d.txt")
🧪 Optional: Train Your Own Word Embeddings
You can train custom Word2Vec or FastText models:

python
Copy
Edit
from gensim.models import Word2Vec

model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format("embeddings/custom_vectors.txt")
📊 Evaluation & Comparison
Embeddings are evaluated by:

Vector visualization (TSNE or PCA)

Similarity checks between words

Performance in downstream tasks (optional)

❗ Common Errors & Fixes
Error	Cause	Fix
FileNotFoundError: glove.6B.100d.txt	GloVe file not in right folder	Place in embeddings/ or correct the path
NameError: tokenizer not defined	Tokenizer is used before it’s created	Make sure to fit Tokenizer first
ValueError: numpy.dtype size changed	Binary version mismatch	pip install --upgrade numpy and restart

📚 References
GloVe Project

Gensim Documentation

TF-IDF – scikit-learn

👩‍💻 Author
Name: Aarti Potdar
Course: NLP / Machine Learning Assignment

