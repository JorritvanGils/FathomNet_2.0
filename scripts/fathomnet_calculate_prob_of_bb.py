import numpy as np


def load_probabilities():
    array = np.loadtxt('/media/jorrit/Storage SSD/fathomnet/Sean/distribution.out', delimiter=",")
    return array

#scaling = 2073 number of pixels in the image (in reality 2073600)
def probability_of_sample(labels, scaling = 2073, array = None):
    if array == None:
        array = load_probabilities()

    prob = []
    for label in labels:
        label = label.split(" ")
        width = float(label[3])
        height = float(label[4])
        surface =  int(width * height * scaling)
        # print('surface', surface)
        if surface > 99: # why first we first fitted untill 999 and now 99?
            surface = 99 # the surface value are the number of pixels the bbox contains, correct?

        # e.g. surface = 2 (box covers 2 pixel)
        # array[2] = 1726 (nr. of training boxes with similar surface value)
        # np.sum(array) -> 839 + 2032 + 1726 + etc. = 23699 (nr of bboxes in the training set(?))

        # array[surface]/ np.sum(array) 
        # 1726/23699 = 0.072
        # probability of the surface area divided over all probabilities 
        # result are extremely small values (They were divided by the total prop,
        # but the prob can never be higher than 2032), so
        # normalising and scaling result between 0 and 1
        # np.max(array) / np.sum(array) 
        # 2032 / 23699 = 0.085
        # (array[surface]/np.sum(array))/(np.max(array)/np.sum(array))
        # 0.072 / 0.085 = 0.857 <- so likely from the distribution
    
        # print('array[surface]', array[surface])
        # print('np.sum(array)', np.sum(array))
        # print('np.max(array)', np.max(array))
        # print('np.sum(array)', np.sum(array))
        # print('(array[surface]/np.sum(array))', array[surface]/np.sum(array))

        prob.append((array[surface]/np.sum(array))/(np.max(array)/np.sum(array)))

    return prob

# [ 839. 2032. 1726. 1496. 1318. 1214. 1127. 1055. 1016.  980.  948.  897.
#   807.  678.  632.  582.  495.  413.  415.  346.  290.  271.  237.  181.
#   174.  142.  133.  144.  152.  120.  125.   91.   88.   79.   75.   66.
#    74.   55.   49.   43.   48.   43.   66.   42.   57.   46.   36.   34.
#    46.   34.   31.   42.   22.   27.   27.   30.   23.   31.   27.   19.
#    31.   18.   28.   18.   28.   16.   15.   23.   23.   18.   21.   13.
#    12.   20.   14.   18.   14.   12.   14.   21.   16.   15.   15.   11.
#    12.   16.   17.   10.   15.    9.   13.    4.   12.    5.    7.    8.
#    11.    8.    7.  775.]