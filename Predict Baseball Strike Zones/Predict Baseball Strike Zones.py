import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
import pprint as pp

fig, ax = plt.subplots()

'''
Data is inside codecademy database and has invisible characters.
Data and code will be executed via Codecademy.com

Project link: https://www.codecademy.com/journeys/data-scientist-ml/paths/dsmlcj-22-machine-learning-ii/tracks/dsmlcj-22-supervised-learning-ii-sv-ms-rm-nb/modules/support-vector-machines-skill-path-fe1996b3-8d55-4eb8-b35b-42b0110f2011-af506b18-77c9-4c9e-85eb-305bc37c29fd/projects/baseball
'''
# Print columns for aaron_judge
#pp.pprint(aaron_judge.columns)

# Print feature descriptions

pp.pprint(aaron_judge.description.unique())