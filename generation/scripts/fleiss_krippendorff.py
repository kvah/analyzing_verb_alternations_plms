from statsmodels.stats import inter_rater as irr
import csv
import numpy as np
import krippendorff as kd

files = ['../generated_data/group_annotations/david_annotations.csv', '../generated_data/group_annotations/james_annotations.csv', 
'../generated_data/group_annotations/jiayu_annotations.csv', '../generated_data/group_annotations/peter_annotations.csv']
def calculate_fleiss_krippendorff():
    ratings = []

    for filename in files:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            curr_ratings = []
            for row in reader:
                curr_ratings.append(int(row[1]))
            ratings.append(curr_ratings)
    
    transposed = np.array(ratings).transpose()
    data, categories = irr.aggregate_raters(transposed)
    print(irr.fleiss_kappa(data, method='fleiss'), kd.alpha(ratings, level_of_measurement='nominal'))

calculate_fleiss_krippendorff()
