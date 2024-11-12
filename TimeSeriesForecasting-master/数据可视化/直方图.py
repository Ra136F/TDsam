# example of a histogram plot
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
# seed the random number generator
seed(1)
# random numbers drawn from a Gaussian distribution
x = randn(10000)
# create histogram plot
pyplot.hist(x)
#pyplot.hist(x,bins=100)
# show line plot
pyplot.show()