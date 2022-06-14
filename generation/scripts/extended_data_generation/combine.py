import csv

outputFile = '../../generated_data/extended_data/all.txt'
with open(outputFile, 'w') as csvfile:
    for file in ['../../generated_data/extended_data/unspecified_extended.tsv', '../../generated_data/extended_data/spray_load_extended.tsv', 
    '../../generated_data/extended_data/dative_extended.tsv', '../../generated_data/extended_data/inch_extended.tsv', 
    '../../generated_data/extended_data/there_extended.tsv']:
    
        #open file, write contents to one txt file
        with open(file, 'r') as curr:
            reader = csv.reader(curr)
            for row in reader:
                for elem in row:
                    csvfile.write(elem + '\t')
                csvfile.write('\n')
    


