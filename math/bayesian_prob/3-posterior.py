#!/usr/bin/env python3
"""shebang setup"""
import numpy as np


def posterior(x, n, P, Pr):
    """function that calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects given the
    data"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or " +
                         "equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    marg = marginal(x, n, P, Pr)

    like = likelihood(x, n, P)

    posterior = (like * Pr) / marg

    return posterior


def marginal(x, n, P, Pr):
    """Calculate marginal probability of obtaining the data

   Parameters:
       x: number of successes (patients with side effects)
       n: number of trials (total patients)
       P: array of probabilities to evaluate
       Pr: array of prior beliefs about P
   Returns:
       marginal probability as a float
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or " +
                         "equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    margprob = likelihood(x, n, P)

    return np.sum(margprob * Pr)


def likelihood(x, n, P):
    """
    Calculate likelihood of observing x
    successes in n trials for different
    probabilities
    Parameters:
    x: number of successes (patients with
    side effects)
    n: number of trials (total patients)
    P: array of probabilities to evaluate
    likelihood for
    Returns:
        array of likelihood values corresponding to each probability in P
    """
    # Check if n is a positive integer
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )

    # Check if x is less than or equal to n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is 1D numpy array
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in [0,1]
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate likelihood using binomial probability formula
    # P(X=x) = C(n,x) * p^x * (1-p)^(n-x)

    # Calculate binomial coefficient C(n,x)
    coef = np.math.factorial(n) / (
        np.math.factorial(x) *
        np.math.factorial(n - x)
    )

    # Calculate p^x and (1-p)^(n-x) for each p in P
    # Then multiply by coefficient
    likelihood = coef * (P**x) * ((1-P)**(n-x))

    return likelihood
