import mlconjug3
import csv
import random

existence_nouns = ["fire", "river", "problem", "plant", "boy", "giant"]
spatial_nouns = ["monkey", "giant", "girl", "bird"]
meander_nouns = ["river", "waterfall", "forest fire", "koala", "tiger"]
appearance_nouns = ["ship", "thunderstorm", "tornado", "hurricane"]
motion_nouns = ["elephant", "horse", "cow", "dog", "cat"]

ending_nouns = ["on the horizon", "in the valley", "behind the mountain", "in the room", "behind me", "above me", "in front of me"]
motion_ending = ["into the air", "into the water", "into the room", "into the sky", "into the ground", "into the ocean", "into the forest", "into the desert"]

##there insertion
there_file = "../POS_data/there_insertion.txt"
lines = open(there_file, 'r').readlines()
there_verbs = {}
curr_verb_type = ""
for line in lines:
    if line.startswith("-"):
        curr_verb_type = line[1:].strip()
        there_verbs[curr_verb_type] = []
        continue
    there_verbs[curr_verb_type].append([x.lower().strip()[:-1] for x in line.split() if len(x) > 0])

for key in there_verbs.keys():
    there_verbs[key] = there_verbs[key][0]

default_conjugator = mlconjug3.Conjugator(language='en')

there_sentences = []

with open('../generated_data/there_sentences.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for verb in there_verbs.keys():
        for verb_type in there_verbs[verb]:
            conjugated_verb = default_conjugator.conjugate(verb_type).conjug_info['indicative']['indicative past tense']['3s']
            sen1, sen2 = "", ""
            if verb == "existence":
                noun = random.choice(existence_nouns)
                sen1 = "A " + noun + " " + conjugated_verb + "."
                sen2 = "There " + conjugated_verb + " a " + noun + "."
            elif verb == "spatial":
                noun = random.choice(spatial_nouns)
                ending = random.choice(ending_nouns)
                sen1 = "A " + noun + " " + conjugated_verb + " " + ending + "."
                sen2 = "There " + conjugated_verb + " a " + noun + " " + ending + "."
            elif verb == "meander":
                noun = random.choice(meander_nouns)
                sen1 = "A " + noun + " " + conjugated_verb + "."
                sen2 = "There " + conjugated_verb + " a " + noun + "."
            elif verb == "appearance" or verb == "disappearance" or verb == "directed":
                noun = random.choice(appearance_nouns)
                ending = random.choice(ending_nouns)
                sen1 = "A " + noun + " " + conjugated_verb + " " + ending + "."
                sen2 = "There " + conjugated_verb + " a " + noun + " " + ending + "."
            elif verb == "manner":
                noun = random.choice(motion_nouns)
                ending = random.choice(motion_ending)
                sen1 = "A " + noun + " " + conjugated_verb + " " + ending + "."
                sen2 = "There " + conjugated_verb + " a " + noun + " " + ending + "."
            else:
                noun = random.choice(existence_nouns + spatial_nouns + meander_nouns + appearance_nouns + motion_nouns)
                ending = random.choice(ending_nouns + motion_ending)
                sen1 = "A " + noun + " " + conjugated_verb + " " + ending + "."
                sen2 = "There " + conjugated_verb + " a " + noun + " " + ending + "."
            there_sentences.append(['there', '0', sen1])
            there_sentences.append(['there', '0', sen2])

    csvwriter.writerows(there_sentences)



