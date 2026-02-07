# E-commerce Text Classification with Traditional ML Models

## Project Overview

This project implements a comparative analysis of traditional machine learning models for e-commerce product text classification. We evaluate multiple embedding techniques (TF-IDF, Word2Vec, GloVe) combined with traditional classifiers to determine optimal approaches for this text classification task.

### Problem Definition

**Objective**: Classify e-commerce product descriptions into their respective categories using traditional machine learning approaches.

**Research Questions**:
1. Which embedding technique provides the best representation for e-commerce text classification?
2. How do different traditional ML models perform across various embeddings?
3. What preprocessing strategies yield optimal results for this domain?

### Dataset

**Source**: [E-commerce Text Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/data)

**Justification**: This dataset is ideal for our comparative embedding analysis because:
- Contains real-world e-commerce product descriptions
- Multi-class classification problem (4 categories)
- Sufficient size for meaningful statistical analysis
- Domain-specific vocabulary suitable for testing embedding approaches

## Project Structure

```
alu/
├── README.md
├── requirements.txt
├── data/
│   └── ecommerceDataset.csv          # Raw dataset (download from Kaggle)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb     # EDA and preprocessing
│   ├── 02_embeddings.ipynb            # Embedding implementations
│   ├── 03_model_training.ipynb        # Model training and evaluation
│   └── 04_results_analysis.ipynb      # Final results and comparison
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Text preprocessing utilities
│   ├── embeddings.py                  # Embedding implementations
│   ├── models.py                      # Model implementations
│   └── evaluation.py                  # Evaluation metrics and utilities
├── models/
│   └── trained/                       # Saved model files
└── results/
    ├── figures/                       # Generated visualizations
    └── tables/                        # Experiment result tables
```

## Embedding Techniques

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Captures term importance across documents
   - Suitable for sparse, high-dimensional representations

2. **Word2Vec (Skip-gram/CBOW)**
   - Dense word embeddings trained on corpus
   - Captures semantic relationships

3. **GloVe (Global Vectors)**
   - Pre-trained embeddings from Stanford
   - Leverages global co-occurrence statistics

## Models Implemented

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes
- XGBoost (Optional)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Dataset Download

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/data)
2. Place `ecommerceDataset.csv` in the `data/` folder

## Usage

Run notebooks in order:
1. `01_eda_preprocessing.ipynb` - Explore data and preprocess
2. `02_embeddings.ipynb` - Generate embeddings
3. `03_model_training.ipynb` - Train and evaluate models
4. `04_results_analysis.ipynb` - Analyze and compare results

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted and macro F1 scores
- **Precision/Recall**: Per-class and averaged
- **Confusion Matrix**: Visual classification performance

## Results Summary

| Model | TF-IDF | Word2Vec | GloVe |
|-------|--------|----------|-------|
| Logistic Regression | - | - | - |
| SVM | - | - | - |
| Random Forest | - | - | - |
| Naive Bayes | - | - | - |

*Results will be populated after running experiments*

## References

1. Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
2. Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
3. Joulin, A., et al. (2016). "Bag of Tricks for Efficient Text Classification"

## Author

[Your Name]

## License

MIT License
# alu-text-classfication
# alu-text-classfication
