#!/usr/bin/env python3
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation: A numpy.ndarray of shape
        (T,) that contains the index of the 
        observation.
        Emission: A numpy.ndarray of shape (N, M) containing the emission 
             probability of a specific observation given a hidden state. 
             Emission[i, j] is the probability of observing j given the hidden 
             state i.
        Transition: A 2D numpy.ndarray of shape (N, N) containing the transition 
                probabilities. Transition[i, j] is the probability of 
                transitioning from the hidden state i to j.
        Initial: A numpy.ndarray of shape (N, 1) containing the probability of 
            starting in a particular hidden state.

        Returns:
        P: The likelihood of the observations given the model.
        B: A numpy.ndarray of shape (N, T) containing the backward path 
            probabilities. B[i, j] is the probability of generating the future 
            observations from hidden state i at time j.
        None, None: On failure.
    """
    conditions = [
        isinstance(Observation, np.ndarray),
        isinstance(Emission, np.ndarray),
        isinstance(Transition, np.ndarray),
        isinstance(Initial, np.ndarray)
    ]

    if not all(conditions):
        return None, None

    conditions = [
        len(Transition.shape) == 2,
        len(Initial.shape) == 2,
        len(Emission.shape) == 2,
        Transition.shape[0] == Transition.shape[1],
        Initial.shape[1] == 1,
        Initial.shape[0] == Transition.shape[0],
        np.allclose(np.sum(Transition, axis=1), 1),
        np.allclose(np.sum(Emission, axis=1), 1),
        np.allclose(np.sum(Initial), 1)
    ]

    if not all(conditions):
        return None, None
    #need clairification on try block.
    try:

        T = Observation.shape[0]
        N = Transition.shape[0]
        B = np.zeros((N, T))
        B[:, -1] = 1

        for t in range(T - 2, -1, -1):
            for n in range(N):
                B[n, t] = np.sum(
                    Transition[n, :] * Emission[:, Observation[t + 1]] *
                    B[:, t + 1]
                )

        P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
