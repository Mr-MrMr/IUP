import os
import re
from nltk import word_tokenize, pos_tag, stem, corpus
import pandas as pd


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


def transform_text(sample):
    # Убираю лишние пробелы
    splitted_sample = " ".join(sample.split())

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


# Создаю датафрейм для хранения обработанных примеров
my_dataframe = pd.DataFrame(columns=["Label", "Text"])
# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", "'m", "'s", "'re", "'ll", "'d", "'ve", "№", "fw"])
# Проверяю каждый файл
for filename in os.listdir("GenSpam_datasets\\Spam"):
    with open(f"GenSpam_datasets\\Spam\\{filename}", 'r') as f:
        file_content = f.read()
    # Мне нужен текст из определённого поля "TEXT_NORMAL",
    # которое идёт после поля "MESSAGE_BODY". Здесь находятся сначала
    # поля "MESSAGE_BODY", а в них уже поля "TEXT_NORMAL"
    text_normal_pattern = re.compile(r'<TEXT_NORMAL>[\s\S]*?</TEXT_NORMAL>')
    message_body_pattern = re.compile(r'<MESSAGE_BODY>[\s\S]*?</MESSAGE_BODY>')
    # Нахожу поля "MESSAGE_BODY"
    message_body_matches = message_body_pattern.findall(file_content)
    for match in message_body_matches:
        # Нахожу поля "TEXT_NORMAL"
        text = " ".join(text_normal_pattern.findall(match))
        # Убираю обозначение поля: "<TEXT_NORMAL>"
        text = text.replace("<TEXT_NORMAL>", "").replace("</TEXT_NORMAL>", "")
        # В датасете имена, названия и проч. личная/опасная информация
        # зацензурены подобным синтаксисом: &WORD. Они не нужны
        censorsed_names = re.compile(r"&[A-Z]+")
        text = censorsed_names.sub(' ', text)
        # Обрабатываю текст
        text = transform_text(text)
        # Если текст имеется, то записываю в датафрейм
        if len(text) > 0:
            my_dataframe.loc[len(my_dataframe)] = [1, " ".join(text)]
for filename in os.listdir("GenSpam_datasets\\Ham"):
    with open(f"GenSpam_datasets\\Ham\\{filename}", 'r') as f:
        file_content = f.read()
    text_normal_pattern = re.compile(r'<TEXT_NORMAL>[\s\S]*?</TEXT_NORMAL>')
    message_body_pattern = re.compile(r'<MESSAGE_BODY>[\s\S]*?</MESSAGE_BODY>')
    message_body_matches = message_body_pattern.findall(file_content)
    for match in message_body_matches:
        text = " ".join(text_normal_pattern.findall(match))
        text = text.replace("<TEXT_NORMAL>", "").replace("</TEXT_NORMAL>", "")
        censorsed_names = re.compile(r"&[A-Z]+")
        text = censorsed_names.sub(' ', text)
        text = transform_text(text)
        if len(text) > 0:
            my_dataframe.loc[len(my_dataframe)] = [0, " ".join(text)]
# Добавляю примеры в csv файл, этот файл будет одним большим датасетом
my_dataframe.to_csv("Final_dataset.csv", mode='a', index=False, header=False)
