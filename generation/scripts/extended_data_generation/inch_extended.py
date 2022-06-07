import csv

rows = []
fava = []
fava_test = []
uniq = []

with open('../../FaVA/inchoative/train.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava.append(row)

with open('../../FaVA/inchoative/test.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava_test.append(row)

with open('../../generated_data/inch_sentences.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

alternator = 0

for row in rows:
    if alternator == 0:
        verb = row[2].split(' ')[1]
    else:
        verb = row[2].split(' ')[-1][:-1]
    alternator = 1 - alternator
    verb_flag = 0
    for line in fava_test:
        if verb == line[3].split(' ')[1]:
            verb_flag == 1
            break
    if verb_flag == 0:
        row[2] = row[2][:-1].lower() + ' .'
        if row[2] not in uniq:
            uniq.append(verb)
            row[2] = "\t" + row[2].strip()
            fava.append(row)



with open('../../generated_data/extended_data/inch_extended.tsv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in fava:
        writer.writerow(row)



