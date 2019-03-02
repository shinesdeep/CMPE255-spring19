from collections import Counter
from linear_algebra import distance
from statistics import mean
import math, random
import matplotlib.pyplot as plt


def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


def predict_preferred_language_by_city(k_values, cities):
    
    for k in k_values:
        num_correct = 0
        for city in cities:
            pred_city = knn_classify(k,cities,city[0])
            if pred_city == city[1]:
                num_correct+=1
        print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))     

if __name__ == "__main__":
    k_values = [1, 3, 5, 7]
    # TODO
    from data import cities
    predict_preferred_language_by_city(k_values, cities)