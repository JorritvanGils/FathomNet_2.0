import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle

def load_brightness_distribution():
    with open('/media/jorrit/Storage SSD/fathomnet/export/train_brightness_list.pkl', 'rb') as f:
        brightness_list = pickle.load(f)
        return brightness_list

def probability_of_sample(brightness_value:float, brightness_list = None) -> float:
    # print(brightness_value)
    if brightness_list == None:
        brightness_list = load_brightness_distribution()
        # Brightness value of 0     = dark
        # Brightness value of 255   = white
        # print(brightness_list)
        # print(len(brightness_list))
        # print(np.mean(brightness_list), np.std(brightness_list), np.min(brightness_list), np.max(brightness_list))
        #       89.66065455996807          31.045161859695845       0.0                      205.91166817223728

    # create (and plot kde)
    kde = stats.gaussian_kde(brightness_list) #gaussian distribution line
    # x = np.linspace(min(brightness_list), max(brightness_list), 100)
    # y = kde.evaluate(x)
    # plt.plot(x, y, color='r', label='KDE')
    # plt.xlabel('Brightness Value')
    # plt.ylabel('Density')
    # plt.title('Gaussian Distribution (KDE)')
    # plt.legend()
    # plt.savefig('/media/jorrit/Storage SSD/fathomnet/export/brightness_gaussian.png')

    # calculate limits.
    left_limit = brightness_value-np.std(brightness_list) # e.g. 90-31.1= 58.9
    right_limit = brightness_value+np.std(brightness_list) # e.g. 90+31.1= 121.1
    
    # calculate area under the kde-curve for these limits
    prob_at_brightness = kde.integrate_box_1d(left_limit, right_limit)
    # print(prob_at_brightness)
    # print(sum(prob_at_brightness))

    # I'm not sure if we need this mechanism
    # normalise the probability to values between 0 and 1 
    # prob_sum = sum(prob_at_brightness)
    # prob_at_brightness = [p/prob_sum for p in prob_at_brightness]

    return(prob_at_brightness)

# # testing
# new_brightness_value_1 = 89.66065455996807
# new_brightness_value_2 = 80
# prob_at_brightness_1 = probability_of_sample(new_brightness_value_1)
# prob_at_brightness_2 = probability_of_sample(new_brightness_value_2)

# print(prob_at_brightness_1, prob_at_brightness_2)



# pdf = kde.evaluate(new_brightness_value)  # translate Gaussian into probabilities (probability density function) for the new value
# print("pdf: ", pdf)

# Calculate the likelihood of the new brightness value belonging to the distribution
#
# likelihood = pdf / kde.integrate_box_1d(new_brightness_value-np.std(brightness_list), new_brightness_value+np.std(brightness_list))  # compare the likelihood with the brightness_distribution.png

# print("Likelihood brightness value belonging to distribution:", likelihood)





