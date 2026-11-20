# 💳 Credit Card Fraud Detection Using FT-Transformer and TabNet

A deep learning project that detects fraudulent credit card transactions using two tabular deep learning architectures: **FT-Transformer** and **TabNet**.

The project compares their performance on a highly imbalanced credit card transaction dataset using classification metrics, validation-based threshold selection, and hyperparameter tuning with Optuna.

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent a very small proportion of all transactions.

A model that predicts every transaction as genuine can achieve high accuracy while failing to detect actual fraud.

The objective of this project is to implement and compare two deep learning models designed for tabular data and evaluate their ability to identify fraudulent transactions.

## 📂 Dataset

**Dataset:** European Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Statistics

| Description | Value |
|---|---:|
| Total transactions | 284,807 |
| Genuine transactions | 284,315 |
| Fraudulent transactions | 492 |
| Input features | 30 |
| Target column | Class |

### Dataset Features

- `Time`: Seconds elapsed since the first transaction.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized features obtained through Principal Component Analysis (PCA).
- `Class`: Target variable (0 = Genuine, 1 = Fraud).

The dataset is highly imbalanced, with approximately 0.173% fraudulent transactions.

## 🧠 Models Implemented

### 1. FT-Transformer

Feature Tokenizer Transformer (FT-Transformer) is a Transformer-based architecture designed for tabular data.

It converts numerical input features into feature tokens and uses self-attention to learn relationships between the features.

The implemented model uses:
- Feature tokenization
- Transformer encoder
- Multi-head self-attention
- Layer normalization
- Classification output layer

### 2. TabNet

TabNet is a deep learning architecture specifically designed for tabular data.

It uses sequential attention mechanisms to select relevant features during different decision steps.

The implementation uses the `pytorch-tabnet` library.

## 🛠️ Technologies Used

- Python
- Google Colab
- PyTorch
- PyTorch TabNet
- Optuna
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## ⚙️ Methodology

The project follows these steps:

1. Load and inspect the credit card transaction dataset.
2. Remove duplicate records and handle infinite values.
3. Separate input features and target labels.
4. Split the dataset into training, validation and testing sets using stratified sampling.
5. Handle missing values using training-set feature medians.
6. Standardize the input features using `StandardScaler`.
7. Train the FT-Transformer model using class-weighted binary cross-entropy loss.
8. Tune FT-Transformer and TabNet hyperparameters using Optuna.
9. Select classification thresholds using validation predictions.
10. Evaluate both models on the held-out test dataset.
11. Compare model performance using multiple evaluation metrics.
12. Generate confusion matrices, ROC curves and Precision-Recall curves.
13. Save the evaluation results and trained model files.

### Data Splitting

The notebook uses stratified sampling with the following proportions:

| Dataset | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

Stratification helps preserve the proportion of genuine and fraudulent transactions across the splits.

### Data Preprocessing

- Infinite values are replaced with missing values.
- Exact duplicate rows are removed.
- Missing numerical values are filled using training-set medians.
- StandardScaler is fitted on training data and applied to validation and test data.

### Hyperparameter Optimization

Optuna is used to search for suitable model hyperparameters.

For FT-Transformer, the search includes:
- Token dimension
- Number of attention heads
- Dropout rate
- Learning rate
- Batch size

For TabNet, the search includes:
- Decision dimension
- Attention dimension
- Number of decision steps
- Learning rate

The current notebook uses a small search of three Optuna trials for each model to keep the experiment relatively lightweight.

## 📊 Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall proportion of correctly classified transactions.
- **Precision:** Proportion of predicted fraud transactions that are actually fraudulent.
- **Recall:** Proportion of actual fraudulent transactions correctly detected.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the ability to distinguish between genuine and fraudulent transactions across classification thresholds.
- **PR-AUC:** Measures precision-recall performance and is particularly useful for highly imbalanced datasets.

PR-AUC is emphasized because fraudulent transactions are extremely rare in this dataset.

## 📈 Experimental Results

The following results were obtained from the uploaded experiment.

### Model Comparison

| Metric | FT-Transformer | TabNet |
|---|---:|---:|
| Accuracy | 99.94% | 99.90% |
| Precision | 88.75% | 75.00% |
| Recall | 74.74% | 56.84% |
| F1-score | 81.14% | 64.67% |
| ROC-AUC | 96.38% | 80.06% |
| PR-AUC | 74.58% | 52.28% |

### Result Interpretation

FT-Transformer achieved better results than TabNet across all reported evaluation metrics in this experiment.

Important observations:

- FT-Transformer achieved higher precision, meaning fewer of its fraud predictions were false alarms.
- FT-Transformer achieved higher recall, detecting a greater proportion of fraudulent transactions.
- FT-Transformer obtained a higher F1-score, indicating a better balance between precision and recall.
- FT-Transformer achieved higher PR-AUC and ROC-AUC values.

Based on this experimental comparison, FT-Transformer performed better than TabNet on the selected test split.

These results represent this particular experimental run and do not establish that FT-Transformer will always outperform TabNet.

## 📉 Visualizations

The notebook generates the following visualizations:

1. Confusion matrices for both models.
2. ROC curves for model comparison.
3. Precision-Recall curves for model comparison.
4. Classification reports showing precision, recall and F1-score.

These visualizations help evaluate the models beyond accuracy alone.

## 💾 Saved Outputs

The notebook generates the following output files:

| File | Description |
|---|---|
| `fraud_detection_results.csv` | Evaluation metrics for both models |
| `ft_transformer_fraud.pth` | Saved FT-Transformer model weights |
| `tabnet_fraud_model.zip` | TabNet model archive generated by its save function |

The TabNet model is saved in the notebook's working directory. Download it separately from Colab if you want to include it in the GitHub repository.

The dataset is downloaded separately from Kaggle and is not included in the repository.

## ⚠️ Limitations

1. The experiment uses a small Optuna search with only three trials per model.
2. FT-Transformer uses class-weighted binary cross-entropy, while TabNet does not explicitly use class-weighted training in the current implementation.
3. The experiment uses a single random seed (42).
4. The results are based on one stratified train-validation-test split.
5. The dataset contains anonymized PCA features, so their original real-world meanings cannot be directly interpreted.
6. No external validation on another credit card dataset or banking system has been performed.
7. The current notebook is a simplified experimental implementation and does not include separate Hyperband or Genetic Algorithm optimization experiments.

## 🏁 Conclusion

This project implements and compares two deep learning architectures, FT-Transformer and TabNet, for credit card fraud detection.

The experimental results show that FT-Transformer achieved better performance than TabNet across the evaluated metrics, including precision, recall, F1-score, ROC-AUC and PR-AUC.

The results demonstrate the potential of Transformer-based architectures for tabular fraud detection while highlighting the challenges associated with highly imbalanced classification datasets.

## 🚀 How to Run the Project

1. Clone or download this GitHub repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload or open the project notebook.
4. Install the required packages using the installation cell.
5. Download `creditcard.csv` from the Kaggle dataset linked above.
6. Run the notebook cells in order.
7. Upload the dataset when prompted.
8. Review the model comparison results and generated visualizations.

## 📁 Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── Simplified_Fraud_Detection_FTTransformer_TabNet.ipynb
├── README.md
├── fraud_detection_results.csv
├── ft_transformer_fraud.pth
└── tabnet_fraud_model.zip
```

*The repository structure should be adjusted to match the files actually uploaded to GitHub.*

## 👨‍💻 Author

Developed as a Deep Learning Lab project focused on comparing FT-Transformer and TabNet architectures for credit card fraud detection.
