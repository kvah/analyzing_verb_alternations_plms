import mlconjug3
import csv
import random

namesList = [["Bob", 0], ["James", 0], ["David", 0], ["Hugh", 0], ["Peter", 0], ["Barbara", 1], ["Sarah", 1], ["Hannah", 1], ["Olga", 1], ["Deandra", 1]] 


##unspecified object
unspecified_object_file = "../POS_data/unspecified_object.txt"
lines = open(unspecified_object_file, 'r').readlines()
unspecified_object_verbs = {}
curr_verb_type = ""
for line in lines:
    if line.startswith("-"):
        curr_verb_type = line[1:].strip()
        unspecified_object_verbs[curr_verb_type] = []
        continue
    if curr_verb_type == "wholebody" or curr_verb_type == "unspecified":
        verbs = line.split()
        for verb in verbs:
            verb = verb.replace(",", "")
            unspecified_object_verbs[curr_verb_type].append(verb.lower().split())
    elif curr_verb_type == "body" or curr_verb_type == "specificbody":
        pairs = line.split(', ')
        verb_and_parts = []
        for pair in pairs:
            elems = pair.split(' ')
            verb = elems[0]
            parts = [x.lower().strip() for x in elems[1:]]
            if verb and parts:
                verb_and_parts.append([verb, parts])
        unspecified_object_verbs[curr_verb_type].append(verb_and_parts)
    else:
        continue

default_conjugator = mlconjug3.Conjugator(language='en')

unspecified_sentences = []

with open('../generated_data/unspecified_object_sentences.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for verb in unspecified_object_verbs.keys():
        for verb_type in unspecified_object_verbs[verb]:
            print(verb_type)
            for curr in verb_type:
                nouns = []
                conjugated_verb = ""
                if type(curr) == 'str':
                    conjugated_verb = default_conjugator.conjugate(str(curr)).conjug_info['indicative']['indicative past tense']['3s']
                else:
                    conjugated_verb = default_conjugator.conjugate(str(curr[0])).conjug_info['indicative']['indicative past tense']['3s']
                    nouns = curr[1:]
                name = random.choice(namesList)
                
                if verb == "unspecified":
                    sen1 = name[0] + " " + conjugated_verb + " a cake."
                    sen2 = name[0] + " " + conjugated_verb + "."
                    unspecified_sentences.append(['u_object', '0', sen1])
                    unspecified_sentences.append(['u_object', '0', sen2])
                
                elif verb == "wholebody":
                    if name[1] == 0:
                        agreement = "himself"
                    else:
                        agreement = "herself"
                    sen1 = name[0] + " " + conjugated_verb + " " + agreement + "."
                    sen2 = name[0] + " " + conjugated_verb + "."
                    unspecified_sentences.append(['u_object', '0', sen1])
                    unspecified_sentences.append(['u_object', '0', sen2])
                
                elif verb == "body":
                    if name[1] == 0:
                        agreement = "his"
                    else:
                        agreement = "her"
                    for word in nouns:
                        for word1 in word:
                            sen1 = name[0] + " " + conjugated_verb + " " + agreement + " " + str(word1) + "."
                            sen2 = name[0] + " " + conjugated_verb + "."
                            unspecified_sentences.append(['u_object', '0', sen1])
                            unspecified_sentences.append(['u_object', '0', sen2])
                else:
                    continue
                
    csvwriter.writerows(unspecified_sentences)



