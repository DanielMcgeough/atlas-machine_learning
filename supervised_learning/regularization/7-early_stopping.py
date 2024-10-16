#!/usr/bin/env python3
"""even in a job I wont stop this i bet."""


def early_stopping(cost, opt_cost, threshold, patience, count):
  """Determines if early stopping should occur.

  Args:
    cost: The current validation cost of the neural network.
    opt_cost: The lowest recorded validation cost of the neural network.
    threshold: The threshold used for early stopping.
    patience: The patience count used for early stopping.
    count: The count of how long the threshold has not been met.

  Returns:
    A boolean   
 indicating whether the network should be stopped early,
    followed by the updated count.
  """

  if cost >= opt_cost - threshold:
    count += 1
  else:
    count = 0
    opt_cost = cost

  stop = count >= patience

  return stop, count
