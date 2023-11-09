import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
from sklearn.model_selection import GridSearchCV
import pprint as pp
import numpy as np

'''
Data is inside codecademy database and has invisible characters.
Data and code will be executed via Codecademy.com

Project link: https://www.codecademy.com/journeys/data-scientist-ml/paths/
dsmlcj-22-machine-learning-ii/tracks/dsmlcj-22-supervised-learning-ii-sv-ms-rm-nb/
modules/support-vector-machines-skill-path-
fe1996b3-8d55-4eb8-b35b-42b0110f2011-af506b18-77c9-4c9e-85eb-305bc37c29fd/projects/baseball
'''

def main():

    fig, ax = plt.subplots()

    def get_strike_zone(player):

        # Change every 'S' to a 1 and every 'B' to a 0.
        player['type'] = player['type'].map({'S':1, 'B':0})

        # Drop NaN values in desired columns
        player = player.dropna(subset=['type', 'plate_x','plate_z'])

        # Plot scatter of pitch positions and outcomes
        plt.scatter(x=player['plate_x'], 
                    y=player['plate_z'], 
                    c=player['type'],
                    cmap=plt.cm.coolwarm, 
                    alpha=0.25)

        # Create SVM to model boundary of real strike zone

        # Split the data
        training_set, validation_set = train_test_split(player, random_state=1)

        # Create SVC model
        classifier = SVC() # Using default rbf kernel

        # Fit the model
        classifier.fit(training_set[['plate_x', 'plate_z']], 
                                    training_set['type'])

        # Visualize SVM
        draw_boundary(ax, classifier)

        # Score classifier
        classifier_score = classifier.score(validation_set[['plate_x', 'plate_z']], 
                                    validation_set['type'])
        print(classifier_score)

        plt.show()


        # Example of overfit model
        classifier = SVC(gamma=100, C=100) # Using default rbf kernel

        # Fit the model
        classifier.fit(training_set[['plate_x', 'plate_z']], 
                                    training_set['type'])

        draw_boundary(ax, classifier)

        # Score classifier
        classifier_score = classifier.score(validation_set[['plate_x', 'plate_z']], 
                                    validation_set['type'])
        print(classifier_score)

        # region Determine optimal values of C and gamma manually

        # # Generate logarithmic values between 0.01 and 100 with more values near 0.01
        # c_range = np.logspace(-2,2,num=50) 
        # gamma_range = np.logspace(-2,2,num=50) 

        # # Create empty list of scores
        # c_accuracy_score = []

        # # Loop through c values and append scores
        # for c in c_range:
        #     classifier = SVC(C=c)
        #     classifier.fit(training_set[['plate_x', 'plate_z']], 
        #                             training_set['type'])
        #     classifier_score = classifier.score(validation_set[['plate_x', 'plate_z']], 
        #                             validation_set['type'])
        #     c_accuracy_score.append(classifier_score)

        # # Get index of highest score
        # best_index = np.argmax(c_accuracy_score)

        # # Get score associated with this index
        # best_C = c_range[best_index]

        # # Create empty list of scores
        # gamma_accuracy_score = []

        # # Loop through c values and append scores
        # for gamma in gamma_range:
        #     classifier = SVC(gamma=gamma)
        #     classifier.fit(training_set[['plate_x', 'plate_z']], 
        #                             training_set['type'])
        #     classifier_score = classifier.score(validation_set[['plate_x', 'plate_z']], 
        #                             validation_set['type'])
        #     gamma_accuracy_score.append(classifier_score)

        # # Get index of highest score
        # best_index = np.argmax(gamma_accuracy_score)

        # # Get score associated with this index
        # best_gamma = gamma_range[best_index]

        # # Create model with best values
        # best_manual_classifier = SVC(C=best_C, gamma=best_gamma)
        # best_manual_classifier.fit(training_set[['plate_x', 'plate_z']], 
        #                             training_set['type'])

        # print(best_manual_classifier.score(validation_set[['plate_x', 'plate_z']], 
        #                             validation_set['type']))

        # Output: 0.8295625942684767

        # endregion

        # Determine optimal model using GridSearchCV
        c_range = np.logspace(-2,2,num=50) 

        parameters = {'kernel':('linear', 'poly', 'rbf'), 
                    'C':c_range,
                    'gamma': ['auto', 'scale']}

        svc = SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(training_set[['plate_x', 'plate_z']], 
                                    training_set['type'])

        # Implement SVC using the best model
        best_model = clf.best_estimator_

        best_model.fit(training_set[['plate_x', 'plate_z']], 
                                    training_set['type'])

        classifier_score = best_model.score(validation_set[['plate_x', 'plate_z']], 
                        validation_set['type'])
        print(classifier_score)

    get_strike_zone(jose_altuve)
    
if __name__ == "__main__":
    main()