from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import speech_recognition as sr
import pandas as pd
import pyaudio
import pickle
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')

# Функция для предобработки текста
def preprocess_text(text, stemmer, stop_words):
    text = re.sub(r'[^а-яА-Я]', ' ', text)
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)

# Загрузка данных из файла output.xlsx
data = pd.read_excel('output.xlsx')
# Предобработка данных
stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words('russian'))
data['processed_question'] = data['question'].apply(lambda x: preprocess_text(x, stemmer, stop_words))
# Разделение на входные и выходные данные
X = data['processed_question']
y = data['answer']
# Преобразование текста в матрицу признаков
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Модель на основе Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Сохранение обученной модели в файл
with open('model_nb.pkl', 'wb') as model_file:
    pickle.dump(model_nb, model_file)
# Загрузка обученной модели из файла
with open('model_nb.pkl', 'rb') as model_file:
    model_nb_loaded = pickle.load(model_file)

for i in range(data.shape[0]):
    # Пример использования загруженной модели
    new_question = data["question"][i]
    print(new_question)
    new_question_processed = preprocess_text(new_question, stemmer, stop_words)
    new_question_count = vectorizer.transform([new_question_processed])
    predicted_answer = model_nb_loaded.predict(new_question_count)
    print(f"Predicted answer: {predicted_answer}")
    # Оценка качества модели на тестовом наборе
    y_pred_test = model_nb_loaded.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy of the model: {accuracy}")
