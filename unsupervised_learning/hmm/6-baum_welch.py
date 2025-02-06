#!/usr/bin/env python3
"""module needed and yes i do like it.
because without it I am"""
import numpy as np

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
  """
  Performs the Baum-Welch algorithm for a hidden Markov model.

  Args:
    Observations: A numpy.ndarray of shape (T,) that contains the index of the 
                 observation.
    Transition: A numpy.ndarray of shape (M, M) that contains the initialized 
                 transition probabilities.
    Emission: A numpy.ndarray of shape (M, N) that contains the initialized 
              emission probabilities.
    Initial: A numpy.ndarray of shape (M, 1) that contains the initialized 
            starting probabilities.
    iterations: The number of times expectation-maximization should be 
                performed. Defaults to 1000.

  Returns:
    The converged Transition, Emission, or None, None on failure.
  """

  try:
    T = len(Observations)
    M = Transition.shape[0]
    N = Emission.shape[1]

    # Validate input shapes
    if Transition.shape != (M, M) or Emission.shape != (M, N) or Initial.shape != (M, 1):
      raise ValueError("Input arrays have incorrect shapes.")

    for _ in range(iterations):
      # Forward-Backward Algorithm
      alpha, beta = forward_backward(Observations, Transition, Emission, Initial)
      gamma = alpha * beta / np.sum(alpha * beta, axis=0) 
      xi = np.zeros((M, M, T-1))
      for t in range(T-1):
        xi[:, :, t] = alpha[:, t][:, np.newaxis] * Transition * Emission[:, Observations[t+1]] * beta[:, t+1]
        xi[:, :, t] /= np.sum(xi[:, :, t])

      # Update parameters
      Initial = gamma[:, 0]
      Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, np.newaxis]
      Emission = np.zeros_like(Emission)
      for j in range(N):
        Emission[:, j] = np.sum(gamma[:, Observations == j], axis=1)
      Emission /= np.sum(gamma, axis=1)[:, np.newaxis]

    return Transition, Emission

  except Exception as e:
    print(f"Error: {e}")
    return None, None

# Helper function for forward-backward algorithm
def forward_backward(Observations, Transition, Emission, Initial):
  T = len(Observations)
  M = Transition.shape[0]

  alpha = np.zeros((M, T))
  beta = np.zeros((M, T))

  # Forward pass
  alpha[:, 0] = Initial.flatten() * Emission[:, Observations[0]]
  for t in range(1, T):
    alpha[:, t] = np.dot(alpha[:, t-1], Transition) * Emission[:, Observations[t]]

  # Backward pass
  beta[:, T-1] = 1
  for t in range(T-2, -1, -1):
    beta[:, t] = np.dot(Transition, Emission[:, Observations[t+1]] * beta[:, t+1])

  return alpha, beta
