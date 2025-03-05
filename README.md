# 🍽️ Restaurant Review Sentiment Analysis 📝

## 📖 Important Note
For a **better understanding** of the project, please check the **Google Colab file** 📄 uploaded in this repository. It contains **detailed explanations and execution steps** to help you grasp the workflow more effectively.

This repository contains a sentiment analysis project using Natural Language Processing (NLP) and a Naive Bayes classifier to classify restaurant reviews as positive 👍 or negative 👎.

## 🔍 Overview
- 📂 The dataset consists of restaurant reviews stored in a TSV file.
- 🧹 Text preprocessing is performed to clean and prepare the data.
- 📊 A Bag of Words (BoW) model is used to convert text data into numerical format.
- 🤖 A Naive Bayes classifier is trained on the dataset to perform sentiment classification.
- 📈 Model evaluation is done using a confusion matrix and accuracy score.

## 🛠️ Technologies Used
- 🐍 Python
- 🗂️ Pandas
- 🔢 NumPy
- 📉 Matplotlib
- 📝 NLTK (Natural Language Toolkit)
- 🤖 Scikit-learn

## ⚙️ Installation
Ensure you have Python installed and set up a virtual environment (optional but recommended).

1. 🚀 Clone this repository:
   ```bash
   git clone https://github.com/yourusername/restaurant-review-nlp.git
   cd restaurant-review-nlp
   ```
2. 📦 Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. 📥 Download the necessary NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## ▶️ Usage
Run the script to preprocess the dataset, train the Naive Bayes model, and evaluate performance:
```bash
python sentiment_analysis.py
```

## 📊 Dataset
The dataset used is `Restaurant_Reviews.tsv`, which contains:
- 🗣️ A column `Review` with customer reviews.
- ✅ A column `Liked` (1 for positive, 0 for negative sentiment).

## 🏗️ Steps in the Code
1. **📥 Load Dataset:** Read the `Restaurant_Reviews.tsv` file.
2. **🧼 Text Cleaning & Preprocessing:**
   - Remove special characters, convert text to lowercase.
   - Remove stopwords (except negations like "not").
   - Apply stemming using `PorterStemmer`.
3. **📊 Feature Extraction:**
   - Use `CountVectorizer` to create a Bag of Words model.
   - Convert text into a numerical matrix representation.
4. **📑 Train-Test Split:**
   - 80% training, 20% testing.
5. **🤖 Train Model:**
   - Train a Multinomial Naive Bayes classifier.
6. **📈 Evaluate Model:**
   - Predict test data.
   - Compute accuracy score and confusion matrix.

## 📊 Model Performance
The script prints:
- 🟩 Confusion matrix for training and test datasets.
- 🎯 Accuracy score of the classifier.

## 🤝 Contribution
Feel free to fork this repository, submit issues, and contribute with improvements! 🚀

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

