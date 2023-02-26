from nltk import stem, corpus, word_tokenize, pos_tag
import pandas as pd

# Определяю csv файл для чтения
FILE = "Enron_spam_data.csv"
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", ",", ".", ":", ";", "\"", "'",
                   "!", "?", "(", ")", "{", "}", "[", "]", "-", "+", "*", "%", "'m", "'s",
                   "'re", "'ll", "'d", "'ve", "@", "#", "№", "$", "^", "&", "=", "~"
                   ">", "<", "`", "fw"])
tokens = []
filtered_tokens = []
# WordNetLemmatizer - это объект, который использует функцию лемматизации, которая ищет слово
# в WordNet (английский словарь для NLTK)
lemmatizer = stem.WordNetLemmatizer()
lemmatized_tokens = []


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


def transform_text(my_data):
    global tokens, filtered_tokens, lemmatized_tokens
    # Создаю токены (разделение текста по словам, предложениям и т.д.)
    for index, row in my_data.iterrows():
        tokens.append(word_tokenize(row['Text']))

    # Лемматизирую каждый токен
    for i in tokens:
        lemmatized_tokens.append(([lemmatizer.lemmatize(token.lower(), pos=penn_to_morphy(tag)) for token, tag in pos_tag(i)]))

    # Фильтрую токены
    for i in lemmatized_tokens:
        filtered_tokens.append([w for w in i if not w.lower() in stop_words])

    return filtered_tokens


# Читаю csv файл и убираю отсутствующие значение (nan)
data = pd.read_csv(FILE, delimiter=',', quotechar='"', encoding='windows-1251', na_values=[])
data['Text'] = data['Text'].fillna('')
# Обрабатываю текст для дальнейшего использования
data["Text"] = transform_text(data)
# Перевожу набор токенов в цельную строку
data['Text'] = data['Text'].agg(lambda x: " ".join(map(str, x)))
# Добавляю примеры в csv файл. Этот датасет я добавлял первым, поэтому я добавляю
# ещё заголовки Label и Text
data.to_csv("Final_dataset.csv", index=False)
