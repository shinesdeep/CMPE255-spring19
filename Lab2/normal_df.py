import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math


def normal_pdf(x, mu=0, sigma=1):
    #implementing the probability distrubution function where mu is mean and sigma is standard deviation
    pi_sqrt = math.sqrt(2 * math.pi)
    return(math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (pi_sqrt * sigma))

           
                   
from matplotlib import pyplot as plt
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '-', label='mu=0,sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], '-', label='mu=0,sigma=0.5')
plt.plot(xs, [normal_pdf(x, sigma=-1) for x in xs], '-', label='mu=0,sigma=-1')
plt.legend()
plt.show()
