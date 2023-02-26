from nltk import word_tokenize, pos_tag, stem, corpus
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import pickle
import re


# Эта функция сейчас тестовая. Она переводит теги в другой формат,
# чтобы эти теги были удобны при лемматизации
def penn_to_morphy(penntag):
    # Конвертирует коды тегов в WordNet
    morphy_tag = {'NN':'n', 'JJ':'a',
                   'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # По умолчанию существительное


def transform_text(input_string):
    # Разбиваю пример на токены (разделение текста по словам, предложениям и т.д.)
    tokens = word_tokenize(input_string)

    # Выдаю теги каждому токему (тег - то, какой частью речи является слово) и
    # лемматизирую каждый токен
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower(), pos=penn_to_morphy(tag))
                         for token, tag in pos_tag(tokens)]
    # Фильтрую токены
    filtered_tokens = [w for w in lemmatized_tokens if not w in stop_words
                           and re.search("^[a-zA-Z0-9]*$", w) != None]
    return filtered_tokens


# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", ",", ".", ":", ";", "\"", "'",
                   "!", "?", "(", ")", "{", "}", "[", "]", "-", "+", "*", "'m", "'s",
                   "'re", "'ll", "'d", "'ve", "@", "#", "№", "$", "^", "&",
                   ">", "<", "fw"])
filtered_tokens = []
# Считываю письмо с ввода пользователя
input_string = input()
# Обрабатываю письмо пользователя
input_string = transform_text(input_string)
input_string = " ".join(input_string)
# Инициализирую векторизатор и векторизирую письмо пользователя
hash_vectorizer = HashingVectorizer()
hash_data = hash_vectorizer.fit_transform([input_string])
# Загружаю классификатор
with open("saved_classifier.pickle", 'rb') as f:
    RF_classifier = pickle.load(f)
# Делаю предсказание
predict_test = RF_classifier.predict_proba(hash_data)
print(predict_test)
# Если вероятность первого больше второго, то это не спам
if predict_test[0][0] > predict_test[0][1]:
    print("ham")
else:
    print("spam")
