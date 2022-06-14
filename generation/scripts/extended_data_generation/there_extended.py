import csv

rows = []
fava = []
fava_test = []
uniq = []

with open('../../FaVA/there/train.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava.append(row)

with open('../../FaVA/there/test.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava_test.append(row)

with open('../../generated_data/there_sentences.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    uniq = []
    for row in reader:
        if row[2] not in uniq:
            uniq.append(row[2])
            rows.append(row)

alternator = 0

for row in rows:
    if alternator == 1:
        verb = row[2].split(' ')[1]
    else:
        verb = row[2].split(' ')[-1][:-1]
    alternator = 1 - alternator
    #check if verb appears in any lines of FaVA
    verb_flag = 0
    alternator2 = 0
    for line in fava_test:
        if alternator2 == 1:
            test_verb = line[3].split(' ')[1]
        else:
            test_verb = line[3].split(' ')[-1][:-1]
        if verb == test_verb:
            verb_flag == 1
            uniq.append(verb)
            break
    if verb_flag == 0:
        row[2] = row[2][:-1].lower() + ' .'
        if verb not in uniq:
            fava.append(row)


with open('../../generated_data/extended_data/there_extended.tsv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in fava:
        uniq = []
        if row[2] not in uniq:
            uniq.append(row[2])
            writer.writerow(row)



