import mlconjug3
import csv
from nltk.corpus import names
import random

namesList = names.words()
random.shuffle(namesList)
namesList = namesList[1:15]
nouns_file = "nouns.txt"
lines = open(nouns_file, 'r').readlines()
nouns = [line.lower().strip() for line in lines]


##causative/inchoative
causative_file = "dative.txt"
lines = open(causative_file, 'r').readlines()
causative_verbs = {}
curr_verb_type = ""
for line in lines:
    if line.startswith("-"):
        curr_verb_type = line[1:].strip()
        causative_verbs[curr_verb_type] = []
        continue
    causative_verbs[curr_verb_type].append([x.lower().strip()[:-1] for x in line.split() if len(x) > 0])

for key in causative_verbs.keys():
    causative_verbs[key] = causative_verbs[key][0]

default_conjugator = mlconjug3.Conjugator(language='en')

inch_sentences = []

with open('dative_sentences.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for verb in causative_verbs.keys():
        for verb_type in causative_verbs[verb]:
            conjugated_verb = default_conjugator.conjugate(verb_type).conjug_info['indicative']['indicative past tense']['3s']
            curr_noun = random.choice(nouns)
            noun2 = random.choice(nouns)
            
            sen1 = random.choice(namesList) + " " + conjugated_verb + " a " + curr_noun + " to " + random.choice(namesList) + "."
            sen2 = random.choice(namesList) + " " + conjugated_verb + " " + random.choice(namesList) + " a " + curr_noun + "."

            inch_sentences.append(['inch', '1', sen1])
            inch_sentences.append(['inch', '0', sen2])

    csvwriter.writerows(inch_sentences)



