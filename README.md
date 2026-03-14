# Fake News Detection (Machine Learning Project)


This project analyzes news articles to distinguish between real and fake news using basic machine learning techniques. The program processes the news text, performs simple preprocessing, and trains classification models to identify whether a news article is real or fake.

The main goal of the project is to demonstrate how machine learning models can be applied to analyze textual data and perform binary classification.

## Dataset

The dataset file **News.csv** contains news articles with the following information:

* **text** – the content of the news article
* **class** – indicates whether the article is real (1) or fake (0)

Other fields such as title, subject, and date are removed during preprocessing because the main analysis focuses on the article text.

## Main Steps

1. Load the news dataset.
2. Remove unnecessary columns such as title, subject, and date.
3. Shuffle the dataset and reset the index.
4. Clean the text by removing punctuation and stopwords.
5. Visualize word patterns using word clouds and frequency charts.
6. Convert text data into numerical features using TF-IDF vectorization.
7. Train machine learning models such as Logistic Regression and Decision Tree.
8. Evaluate the models using accuracy scores and a confusion matrix.

## Technologies Used

Python, Pandas, Matplotlib, Seaborn, NLTK, WordCloud, Scikit-learn.

## Project Structure

Fake_News_Detection/

fake_news_detection.py
News.csv

## Running the Project

Install the required libraries:

pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn tqdm

Then run the script:

python fake_news_detection.py

The program will process the dataset, train machine learning models, and display evaluation results and visualizations.
