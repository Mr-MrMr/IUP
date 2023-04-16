from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Определяю файл с набором данных
FILE = "Final_dataset_4.csv"
# Читаю csv файл и убираю отсутствующие значение (nan)
data = pd.read_csv(FILE, delimiter=',', quotechar='"', encoding='utf-8', na_values=[])
data['Text'] = data['Text'].fillna('')
# Инициализирую векторизатор и векторизирую текст
hash_vectorizer = HashingVectorizer()
hash_data = hash_vectorizer.fit_transform(data["Text"])
# Разделяю рандомно данные на тренировочные и тестовые
text_train, text_test, label_train, label_test = train_test_split(
    hash_data, data['Label'], test_size=0.3
)
# Инициализирую и тренирую классификатор
RF_classifier = RandomForestClassifier()
RF_classifier.fit(text_train, label_train)
# Делаю предсказания
predict_train = RF_classifier.predict(text_train)
print(f"Accuracy of Train dataset {accuracy_score(label_train, predict_train)}")
predict_test = RF_classifier.predict(text_test)
print(f"Accuracy of Test dataset {accuracy_score(label_test, predict_test)}")
# Сохраняю классификатор
with open("saved_classifier_4.pickle", 'wb') as f:
    pickle.dump(RF_classifier, f)
