#!/usr/bin/env python3
"""module for bayesian optimization"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that intializes and runs bayesian optimization
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian optimization.

        Args:
            f:          function to be optimized
            X_init:     Initial X inputs matrix of shape (t, 1)
            Y_init:     Initial Y outputs matrix of shape (t, 1)
            bounds:     Tuple of (min, max) fur searching optimal point
            ac_samples: Number of samples fur acquisition analysis
            l:          Length param fur the kernel (default=1)
            sigma_f:    Standard deviation fur black-box function (default=1)
            xsi:        Exploration-exploitation factor (default=0.01)
            minimize:   True fur minimization, False fur maximization
                        (default=True)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # Generate acquisition sample points
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound,
                               max_bound,
                               ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculations fur next best sample location using
        Expected Improvement (EI).

        Returns:
            tuple: X_next (1,) array of next best point,
                  EI (ac_samples,) array of expected improvements
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Get current best depending on minimization/maximization
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        # Needed fur stability
        sigma = np.maximum(sigma, 1e-9)

        # Calculate improvement based on optimization type
        with np.errstate(divide='warn'):
            if self.minimize:
                imp = mu_sample_opt - mu - self.xsi
            else:
                imp = mu - mu_sample_opt - self.xsi

            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Set EI to 0 where sigma is 0
            ei[sigma == 0] = 0

        # Find index of best EI
        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimize that black-box function
        The exact mathematical relationship between hyperparameters
        and model performance is usually unknown.
        It's a "black box" because:
        We can't directly calculate the optimal
        hyperparameter values.
        We don't have a formula to predict the performance
        for every possible combination.
        We can only observe the output (model performance)
        for a given input (hyperparameter settings).

        Args:
            iterations: Max number of iterations 2 perform (default=100)

        Returns:
            tuple: X_opt (1,) optimal point array,
                  Y_opt (1,) optimal value array
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Check if point was already sampled
            if np.any(np.abs(X_next - self.gp.X) <= 1e-10):
                break

            # Sample new point and update GP
            Y_next = self.f(X_next)
            self.gp.update(X_next.reshape(-1, 1), Y_next.reshape(-1, 1))

        # Get optimal point based on minimization/maximization
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
