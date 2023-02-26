import os
import csv

# Создаю файл
header = ["Label", "Text"]
FILE = open("Enron_spam_data.csv", 'w', newline="")
# Инициализирую писателя в csv файл
csvwriter = csv.writer(FILE)
csvwriter.writerow(header)

# Проверяю каждый файл (1 файл - 1 пример)
for filename in os.listdir("Enron_datasets\\enron1\\spam"):
    try:
        # Читаю весь файл
        line = open(f"Enron_datasets\\enron1\\spam\\{filename}", "r").read()
        # Записываю в файл всё письмо
        csvwriter.writerows([['1', " ".join(line.split()[1:])]])
    except UnicodeDecodeError:
        pass
for filename in os.listdir("Enron_datasets\\enron1\\ham"):
    try:
        # Читаю весь файл
        line = open(f"Enron_datasets\\enron1\\ham\\{filename}").read()
        # Записываю в файл всё письмо
        csvwriter.writerows([['0', " ".join(line.split()[1:])]])
    except UnicodeDecodeError:
        pass
