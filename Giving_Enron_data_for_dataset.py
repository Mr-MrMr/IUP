from nltk import word_tokenize, pos_tag, stem, corpus
import re
import os
import csv


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
    # Удаление ссылок
    sample = " ".join([x for x in sample.split() if
                                   re.search(r"(?:(?:https?|ftp|sftp|file|git|html|font|url):\/\/)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,63}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)", x) == None])

    # Удаляю всё, что не является цифрой, английской буквой или пробелом
    delete_all_non_alphabet_or_numeral = re.compile(r"[^a-zA-Z0-9 ]")
    sample = delete_all_non_alphabet_or_numeral.sub(' ', sample)

    # Разбиваю пример на токены (разделение текста по словам, предложениям и т.д.)
    tokens = word_tokenize(sample)

    # Выдаю теги каждому токему (тег - то, какой частью речи является слово) и
    # лемматизирую каждый токен
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower(), pos=penn_to_WordNet(tag))
                         for token, tag in pos_tag(tokens)]

    # Фильтрую токены
    filtered_tokens = [w.lower() for w in lemmatized_tokens
                       if not w.lower() in stop_words]
    return filtered_tokens


# Создаю файл
header = ["Label", "Text"]
FILE = open("Final_dataset.csv", 'w', newline="")
# Инициализация писателя в csv файл
csvwriter = csv.writer(FILE)
# Этот датасет я добавлял первым, поэтому я добавляю
# ещё заголовки Label и Text
csvwriter.writerow(header)
# Инициализация лемматизатора
lemmatizer = stem.WordNetLemmatizer()
# Создаю множество стоп-слов и знаков препинания
stop_words = set(corpus.stopwords.words('english'))
stop_words.update(["n't", "'m", "'s", "'re", "'ll", "'d", "'ve", "№", "fw"])


# Проверяю каждый файл (1 файл - 1 пример)
for filename in os.listdir("Enron_datasets\\enron1\\spam"):
    try:
        # Читаю весь файл
        line = open(f"Enron_datasets\\enron1\\spam\\{filename}", "r").read()
        # Обрабатываю пример
        line = transform_text(line)
        # Записываю в файл всё письмо
        csvwriter.writerows([['1', " ".join(line[1:])]])
    except UnicodeDecodeError:
        pass
for filename in os.listdir("Enron_datasets\\enron1\\ham"):
    try:
        # Читаю весь файл
        line = open(f"Enron_datasets\\enron1\\ham\\{filename}").read()
        # Обрабатываю пример
        line = transform_text(line)
        # Записываю в файл всё письмо
        csvwriter.writerows([['0', " ".join(line[1:])]])
    except UnicodeDecodeError:
        pass
