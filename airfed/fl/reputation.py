"""Reputation scores for federated partners (paper Eqs. 77-80).

A UAV trusts a neighbour more when it
  * finishes the tasks it runs before their deadlines   (Succ, Eq. 77)
  * reliably delivers its model during FL exchanges      (Stab, Eq. 78)
"""

import numpy as np


class Reputation:
    def __init__(self, n_uavs, alpha_succ=0.6, alpha_stab=0.4, rho=0.75):
        self.alpha_succ, self.alpha_stab, self.rho = alpha_succ, alpha_stab, rho
        self.score = np.ones(n_uavs)                 # everyone starts fully trusted
        self.attempts = np.zeros(n_uavs)
        self.delivered = np.zeros(n_uavs)

    def record_exchange(self, sender, ok):
        self.attempts[sender] += 1
        self.delivered[sender] += ok

    def update(self, exec_total, exec_met):
        """Blend reliability and stability, then smooth with an EMA (Eqs. 79-80)."""
        succ = np.where(exec_total > 0, exec_met / np.maximum(exec_total, 1), 1.0)
        stab = np.where(self.attempts > 0, self.delivered / np.maximum(self.attempts, 1), 1.0)
        fresh = self.alpha_succ * succ + self.alpha_stab * stab
        self.score = self.rho * self.score + (1 - self.rho) * fresh
        return self.score
