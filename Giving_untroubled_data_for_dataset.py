import os
import pandas as pd
from email.policy import default
from email import message_from_string
from nltk import word_tokenize, pos_tag, stem, corpus
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


def transform_text(sample):
    # Разбиваю пример на токены (разделение текста по словам, предложениям и т.д.)
    tokens = word_tokenize(sample)

    # Выдаю теги каждому токему (тег - то, какой частью речи является слово) и
    # лемматизирую каждый токен
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower(), pos=penn_to_morphy(tag))
                         for token, tag in pos_tag(tokens)]

    # Фильтрую токены
    filtered_tokens = [w.lower() for w in lemmatized_tokens
                       if not w.lower() in stop_words]
    return filtered_tokens


# Создаю датафрейм для хранения обработанных примеров
my_dataframe = pd.DataFrame(columns=["Label", "Text"])
# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", ",", ".", ":", ";", "\"", "'",
                   "!", "?", "(", ")", "{", "}", "[", "]", "-", "+", "*", "%", "'m", "'s",
                   "'re", "'ll", "'d", "'ve", "@", "#", "№", "$", "^", "&", "=", "~"
                   ">", "<", "`", "fw"])

# Проверяю каждый файл (1 файл - 1 пример)
for filename in os.listdir("2021\\01"):
    try:
        # Читаю файл с примером
        with open(f"2021\\01\\{filename}") as f:
            sample = f.read()
        # Меняю неизвестные кодировки известной
        sample = sample.replace("charset=\"DEFAULT_CHARSET\"", "charset=\"Windows-1252\"")
        sample = sample.replace("charset=unknown-8bit", "charset=\"Windows-1252\"")
        sample = sample.replace("charset=\"DEFAULT\"", "charset=\"Windows-1252\"")
        sample = sample.replace("charset=\"GB2312_CHARSET\"", "charset=\"Windows-1252\"")
        sample = sample.replace("charset=\"_iso-2022-jp$ESC\"", "charset=\"Windows-1252\"")
        # Получаю тело письма
        msg = message_from_string(sample, policy=default)
        body = msg.get_body(('plain'))
        # Если письмо подходит по содержанию
        if body != None:
            # Получаю основной текст письма
            body = body.get_content()
            # Обрабатываю текст
            body = transform_text(body)
            # Ссылки после обработки разделяются на несколько частей,
            # поэтому нужно убрать эти части
            body = [x for x in body if
                    re.search("[\s\S]*html|font|url|http|~|=|\\\[\s\S]*", x) == None]
            body = [w for w in body if not w in stop_words]
            # Если текст имеется, то записываю в датафрейм
            if len(body) > 0:
                my_dataframe.loc[len(my_dataframe)] = [1, body]
    except UnicodeDecodeError:
        pass

# Перевожу список слов в целую строку для каждого примера
my_dataframe["Text"] = my_dataframe["Text"].agg(lambda x: " ".join(map(str, x)))
# Добавляю примеры в csv файл, этот файл будет одним большим датасетом
my_dataframe.to_csv("Final_dataset.csv", mode='a', index=False, header=False)
