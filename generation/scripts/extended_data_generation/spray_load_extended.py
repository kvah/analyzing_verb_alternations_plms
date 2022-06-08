import csv

rows = []
fava = []
fava_test = []
uniq = []

with open('../../FaVA/spray_load/train.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava.append(row)

with open('../../FaVA/spray_load/test.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    for row in reader:
        fava_test.append(row)

with open('../../generated_data/spray_load_sentences.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

for row in rows:
    verb = row[2].split(' ')[1]
    #check if verb appears in any lines of FaVA
    verb_flag = 0
    for line in fava_test:
        if verb == line[3].split(' ')[1]:
            verb_flag == 1
            break
    if verb_flag == 0:
        row[2] = row[2][:-1].lower() + ' .'
        if row[2] not in uniq:
            uniq.append(row[2])
            fava.append(row)

with open('../../generated_data/extended_data/spray_load_extended.tsv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter="\t", quotechar='"')
    for row in fava:
        writer.writerow(row)


