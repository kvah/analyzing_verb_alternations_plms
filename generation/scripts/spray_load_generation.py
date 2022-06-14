import mlconjug3
import csv
from nltk.corpus import names
import random

namesList = names.words()
random.shuffle(namesList)
namesList = namesList[1:15]
nouns_object = ["paint", "water", "juice", "oil", "cream cheese"]
nouns_loc = ["wall", "floor", "door", "window", "table", "chair", "couch", "bed", "desk"]



##spray load
spray_load_file = "../POS_data/spray_load.txt"
lines = open(spray_load_file, 'r').readlines()
spray_load_verbs = {}
curr_verb_type = ""
for line in lines:
    if line.startswith("-"):
        curr_verb_type = line[1:].strip()
        spray_load_verbs[curr_verb_type] = []
        continue
    spray_load_verbs[curr_verb_type].append([x.lower().strip()[:-1] for x in line.split() if len(x) > 0])

for key in spray_load_verbs.keys():
    spray_load_verbs[key] = spray_load_verbs[key][0]

default_conjugator = mlconjug3.Conjugator(language='en')

loc_sentences = []

with open('../generated_data/spray_load_sentences.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for verb in spray_load_verbs.keys():
        for verb_type in spray_load_verbs[verb]:
            conjugated_verb = default_conjugator.conjugate(verb_type).conjug_info['indicative']['indicative past tense']['3s']
            curr_noun = random.choice(nouns_object)
            noun2 = random.choice(nouns_loc)
            name = random.choice(namesList)
            sen1 = name + " " + conjugated_verb + " " + curr_noun + " on the " + noun2 + "."
            sen2 = name + " " + conjugated_verb + " the " + noun2 + " with " + curr_noun + "." 

            loc_sentences.append(['spray', '0', sen1])
            loc_sentences.append(['spray', '0', sen2])
    csvwriter.writerows(loc_sentences)



