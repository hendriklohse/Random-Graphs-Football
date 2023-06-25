"""
The scipy stats package has great functionality for probability distributions
and random variables, but its random number generator is extremely slow
when one samples only one random number at a time. It is extremely fast,
however, when one samples multiple random numbers simultaneously.
For this reason, we have created this class that acts somewhat as a wrapper
around the scipy probability distributions. It will make sure that random
numbers are always generated in batches of n = 10000, and the rvs() function
simply returns the next random number from this list (and resamples when necessary).

@author: Marko Boon
"""

from scipy import stats

class Distribution:
    n = 100000  # standard random numbers to generate

    '''
    Constructor for this Distribution class.

    Args:
            dist (scipy.stats random variable): A random variable from the scipy stats library.

    Attributes:
            dist (scipy.stats random variable): A random variable from the scipy stats library.
            n (int): a number indicating how many random numbers should be generated in one batch
            randomNumbers: a list of n random numbers generated from 'dist'
            idx (int): a number keeping track of how many random numbers have been sampled

    '''

    def __init__(self, dist):
        self.dist = dist
        self.resample()

    def resample(self):
        self.randomNumbers = self.dist.rvs(size = self.n)
        self.idx = 0

    def rvs(self, n_=1):
        """
        A function that returns n (=1 by default) random numbers from the specified distribution.

        Returns:
            One random number (float) if n=1, and a list of n random numbers otherwise.
        """
        if self.idx >= self.n - n_:
            while n_ > self.n:
                self.n *= 10
            self.resample()
        if n_ == 1:
            # print(self.randomNumbers)
            # print(type(self.randomNumbers))
            # print(self.idx)
            # print(type(self.idx))
            rs = self.randomNumbers[self.idx]
        else:
            rs = self.randomNumbers[self.idx:(self.idx + n_)]
        self.idx += n_
        # print("index", str(self.idx))
        return rs

    def rvs_1(self):
        """
        A function that returns n (=1 by default) random numbers from the specified distribution.

        Returns:
            One random number (float) if n=1, and a list of n random numbers otherwise.
        """
        if self.idx >= self.n - 1:
            while 1 > self.n:
                self.n *= 10
            self.resample()
        rs = self.randomNumbers[self.idx]
        self.idx += 1
        return rs



