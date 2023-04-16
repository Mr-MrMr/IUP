from nltk import word_tokenize, pos_tag, stem, corpus
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import re


# Эта функция переводит теги в другой формат,
# чтобы эти теги были удобны при лемматизации
def penn_to_WordNet(penn_treebank_tag):
    # Конвертирую коды тегов в WordNet
    WordNet_tag = {'N':'n', 'J':'a',
                   'V':'v', 'R':'r'}
    try:
        return WordNet_tag[penn_treebank_tag[0]]
    except:
        return 'n' # По умолчанию существительное


def transform_text(input_string):
    # Разбиваю по пробелам, чтобы убрать ссылки
    splitted_sample = input_string.split()
    splitted_sample = " ".join([x for x in splitted_sample
                                if re.search(r"(?:(?:https?|ftp|sftp|file|git|html|font|url):\/\/)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,63}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)", x) == None])

    # Удаляю всё, что не является цифрой, английской буквой или пробелом
    delete_all_non_alphabet_or_numeral = re.compile(r"[^a-zA-Z0-9 ]")
    splitted_sample = delete_all_non_alphabet_or_numeral.sub(' ', splitted_sample)

    # Разбиваю пример на токены (разделение текста по словам, предложениям и т.д.)
    tokens = word_tokenize(splitted_sample)

    # Выдаю теги каждому токему (тег - то, какой частью речи является слово) и
    # лемматизирую каждый токен
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower(), pos=penn_to_WordNet(tag))
                         for token, tag in pos_tag(tokens)]
    # Фильтрую токены
    filtered_tokens = [w.lower() for w in lemmatized_tokens
                       if not w.lower() in stop_words]
    return filtered_tokens


# Считываю письмо с ввода пользователя
input_string = input()
# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", "'m", "'s", "'re", "'ll", "'d", "'ve", "№", "fw"])
# Загружаю классификатор
with open("saved_classifier.pickle", 'rb') as f:
    RF_classifier = pickle.load(f)
# Инициализирую векторизатор
hash_vectorizer = HashingVectorizer()
while input_string != 'exit':
    # Обрабатываю письмо пользователя
    input_string = transform_text(input_string)
    input_string = " ".join(input_string)
    # Векторизирую письмо пользователя
    hash_data = hash_vectorizer.fit_transform([input_string])
    # Делаю предсказание
    predict_test = RF_classifier.predict_proba(hash_data)
    print(predict_test)
    # Если вероятность первого больше второго, то это не спам
    if predict_test[0][0] > predict_test[0][1]:
        print("legal")
    else:
        print("spam")
    # Считываю письмо с ввода пользователя снова
    input_string = input()
