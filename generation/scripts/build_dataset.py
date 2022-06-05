import csv
import random

filenames = ['../generated_data/spray_load_sentences.csv', '../generated_data/unspecified_object_sentences.csv', 
'../generated_data/there_sentences.csv', '../generated_data/dative_sentences.csv', '../generated_data/inch_sentences.csv']

allLines = []
combined = []

with open("../generated_data/model_files/all.txt", "w") as outfile:
    for filename in filenames:
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                cleaned = row[2].lower().replace(".", " .")
                line = row[0] + "\t" + row[1] + "\t\t" + cleaned + "\n"
                outfile.write(line)
                allLines.append(line)

combined = allLines

def train_dev_test_split(allLines, train_size, dev_size, test_size):
    random.shuffle(allLines)
    train_lines = allLines[:int(train_size*len(allLines))]
    dev_lines = allLines[int(train_size*len(allLines)):int((train_size+dev_size)*len(allLines))]
    test_lines = allLines[int((train_size+dev_size)*len(allLines)):]
    return train_lines, dev_lines, test_lines

#create train, dev, and test files
train_lines, dev_lines, test_lines = train_dev_test_split(allLines, 0.85, 0.05, 0.1)

#write files to txt
with open("../generated_data/model_files/train.txt", "w") as outfile:
    for line in train_lines:
        outfile.write(line)
with open("../generated_data/model_files/dev.txt", "w") as outfile:
    for line in dev_lines:
        outfile.write(line)
with open("../generated_data/model_files/test.txt", "w") as outfile:
    for line in test_lines:
        outfile.write(line)

with open("../generated_data/all.txt", "r") as old_dataset:
    lines = old_dataset.readlines()
    for line in lines:
        if line:
            combined.append(line)

with open("../generated_data/model_files/combined_all.txt", "w") as outfile:
    for line in combined:
        outfile.write(line)

train_lines, dev_lines, test_lines = train_dev_test_split(combined, 0.85, 0.05, 0.1)

with (open("../generated_data/model_files/combined_train.txt", "w")) as outfile:
    for line in train_lines:
        outfile.write(line)
with (open("../generated_data/model_files/combined_dev.txt", "w")) as outfile:
    for line in dev_lines:
        outfile.write(line)
with (open("../generated_data/model_files/combined_test.txt", "w")) as outfile:
    for line in test_lines:
        outfile.write(line)