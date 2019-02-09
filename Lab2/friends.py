num_friends = [100, 49, 41, 40, 25, 10, 5, 4, 7, 20, 60]
def mean_num_friends(x):
    return sum(x)/len(x)

def median_num_friends(x):
    x_sorted=sorted(x)
    middle = float(len(x))/2
    if middle % 2 != 0:
        return x_sorted[int(middle - .5)]
    else:
        return (x_sorted[int(middle)], x_sorted[int(middle-1)])

meanx=mean_num_friends(num_friends)
medy= median_num_friends(num_friends)
print("Mean of the list is:",mean_num_friends(num_friends))
print("Median of the list is:", median_num_friends(num_friends)) 

