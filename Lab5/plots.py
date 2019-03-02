from collections import Counter
from linear_algebra import distance
from statistics import mean
import math, random
import matplotlib.pyplot as plt

from knn import knn_classify

def plot_state_borders(plt, color='0.8'):
    pass

def plot_cities(cities):

    # key is language, value is pair (longitudes, latitudes)
    plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }

    # we want each language to have a different marker and color
    markers = { "Java" : "o", "Python" : "s", "R" : "^" }
    colors  = { "Java" : "r", "Python" : "b", "R" : "g" }

    for (longitude, latitude), language in cities:
        plots[language][0].append(longitude)
        plots[language][1].append(latitude)

    # create a scatter series for each language
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                          label=language, zorder=10)

    plot_state_borders(plt)    # assume we have a function that does this

    plt.legend(loc=0)          # let matplotlib choose the location
    plt.axis([-130,-60,20,55]) # set the axes
    plt.title("Favorite Programming Languages")
    plt.show()


def classify_and_plot_grid(cities, k=1):
    
    
    plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }
    markers = { "Java" : "o", "Python" : "s", "R" : "^" }
    colors  = { "Java" : "r", "Python" : "b", "R" : "g" }

    
    
    for city in cities:

        pred_lang = knn_classify(k,cities, city[0])
        plots[pred_lang][0].append(city[0][0])
        plots[pred_lang][1].append(city[0][1])


    # create a scatter series for each language
    # See above plot_cities() to plot your prediction.
    for language, (x, y) in plots.items():
       plt.scatter(x, y, color=colors[language], marker=markers[language],
                          label=language, zorder=10)

    plot_state_borders(plt) 
    plt.legend(loc=0)          # let matplotlib choose the location
    plt.axis([-130,-60,20,55]) # set the axes
    plt.title("Predicted Programming Languages for k value of %d" %(k))
    plt.show()
      


if __name__ == "__main__":
   
    cities = []
    from data import cities
    plot_cities(cities)
    classify_and_plot_grid(cities)
    classify_and_plot_grid(cities, 3)
    classify_and_plot_grid(cities, 5)
    
    
