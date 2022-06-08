import mlconjug3
import csv
from nltk.corpus import names
import random

namesList = names.words()
random.shuffle(namesList)
namesList = namesList[1:15]
nouns_file = "../POS_data/nouns.txt"
lines = open(nouns_file, 'r').readlines()
nouns = [line.lower().strip() for line in lines]


##causative/inchoative
causative_file = "../POS_data/causative_inchoative.txt"
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

with open('../generated_data/inch_sentences.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for verb in causative_verbs.keys():
        for verb_type in causative_verbs[verb]:
            conjugated_verb = default_conjugator.conjugate(verb_type).conjug_info['indicative']['indicative past tense']['3s']
            curr_noun = random.choice(nouns)
            noun2 = random.choice(nouns)
            if verb == "give" or verb == "contribute" or verb == "future":
                sen1 = random.choice(["they", random.choice(namesList)]) + " " + conjugated_verb + " the " + curr_noun + " to me."
                sen2 = "The " + curr_noun + " " + conjugated_verb + " to " + random.choice(namesList) + "."
            else:
                sen1 = "The " + curr_noun + " " + conjugated_verb + "."
                sen2 = random.choice(namesList) + " " + conjugated_verb + " the " + curr_noun + "."

            inch_sentences.append(['inch', '1', sen1])
            inch_sentences.append(['inch', '0', sen2])

    csvwriter.writerows(inch_sentences)



