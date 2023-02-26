import pandas as pd
from nltk import word_tokenize, pos_tag, stem, corpus


# Определяю csv файл для чтения
FILE = "spam_SMS.csv"
data = pd.read_csv(FILE, delimiter=',', quotechar='"', encoding='windows-1251', na_values=[])
# Убираю лишние столбики
data.drop("Unnamed: 2", inplace=True, axis=1)
data.drop("Unnamed: 3", inplace=True, axis=1)
data.drop("Unnamed: 4", inplace=True, axis=1)
# Убираю пустые значения (NaN - not a number)
data["v2"] = data["v2"].fillna('')
# Меняю названия для удобства
data = data.rename(columns={"v1": "Label", "v2": "Text"})
data["Label"] = data["Label"].replace("ham", 0)
data["Label"] = data["Label"].replace("spam", 1)
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", ",", ".", ":", ";", "\"", "'",
                   "!", "?", "(", ")", "{", "}", "[", "]", "-", "+", "*", "'m", "'s",
                   "'re", "'ll", "'d", "'ve", "@", "#", "№", "$", "^", "&",
                   ">", "<", "fw"])
# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
lemmatized_tokens = []
tokens = []
filtered_tokens = []


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
    # Разбиваю пример на токены (разделение текста по словам, предложениям и т.д.)
    for index, row in my_data.iterrows():
        tokens.append(word_tokenize(row['Text']))

    # Выдаю теги каждому токему (тег - то, какой частью речи является слово) и
    # лемматизирую каждый токен
    for i in tokens:
        lemmatized_tokens.append(([lemmatizer.lemmatize(token.lower(),
                    pos=penn_to_morphy(tag)) for token, tag in pos_tag(i)]))

    # Фильтрую токены
    for i in lemmatized_tokens:
        filtered_tokens.append([w for w in i if not w.lower() in stop_words
                                and not '.' in list(w)])
    return filtered_tokens


# Обрабатываю примеры
data["Text"] = transform_text(data)
# Перевожу список слов в целую строку для каждого примера
data['Text'] = data['Text'].agg(lambda x: " ".join(map(str, x)))
# Добавляю примеры к csv файлу, этот файл будет одним большим датасетом
data.to_csv("Final_dataset.csv", mode='a', index=False, header=False)
