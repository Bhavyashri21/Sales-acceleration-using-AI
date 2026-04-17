"""
hawkes_model.py
Implements a simplified profile-specific Hawkes process win-propensity scorer.
Based on: Yan et al. (2015) "On Machine Learning towards Predictive Sales Pipeline Analytics"

Key ideas from the paper implemented here:
  - Exponential decay kernel: g(t) = w * exp(-w * t)
  - Self-exciting interaction intensity (u=1)
  - Outcome intensity influenced by interactions (u=2)
  - Profile-specific base intensity via logistic function
"""

import numpy as np
from scipy.optimize import minimize


def exponential_kernel(delta_t: float, w: float) -> float:
    """g(t) = w * exp(-w * t)  [Eq. from paper, Section: Seller-pipeline Interaction Modeling]"""
    return w * np.exp(-w * delta_t)


def kernel_integral(delta_t: float, w: float) -> float:
    """G(t) = integral of g from 0 to t = 1 - exp(-w*t)"""
    return 1.0 - np.exp(-w * delta_t)


def logistic(theta: np.ndarray, x: np.ndarray) -> float:
    """Profile-specific base intensity: mu_s = mu0 / (1 + exp(-theta^T x))"""
    return 1.0 / (1.0 + np.exp(-np.clip(np.dot(theta, x), -20, 20)))


def compute_interaction_intensity(times: list, t: float, mu: float, a: float, w: float) -> float:
    """Conditional intensity lambda(t) for interaction process at time t."""
    exciting = sum(
        a * exponential_kernel(t - ti, w)
        for ti in times if ti < t
    )
    return mu + exciting


def hawkes_log_likelihood_u1(times: list, T: float, mu: float, a: float, w: float) -> float:
    """
    Log-likelihood for self-exciting interaction process (u=1).
    L1 = sum_i log(lambda(ti)) - integral_0^T lambda(t)dt
    """
    if len(times) == 0:
        return -T * mu

    ll = 0.0
    for i, ti in enumerate(times):
        lam = mu + sum(a * exponential_kernel(ti - tj, w) for tj in times[:i])
        ll += np.log(max(lam, 1e-10))

    # Integral term
    integral = T * mu + a * sum(kernel_integral(T - tj, w) for tj in times)
    return ll - integral


def win_propensity_score(times: list, T: float, mu2: float, a21: float, w21: float,
                         window: float = 14.0) -> float:
    """
    Win propensity = integral of win intensity over next [T, T+window] days.
    lambda_win(t) = mu2 + sum_j a21 * g(t - tj)
    Approximated as: mu2*window + a21 * sum_j G(window) * exp(-w21*(T-tj))
    """
    base_win = mu2 * window
    exciting_win = 0.0
    for tj in times:
        if tj <= T:
            exciting_win += a21 * kernel_integral(window, w21) * np.exp(-w21 * (T - tj))
    raw = base_win + exciting_win
    # Sigmoid to get probability in [0,1]
    return 1.0 / (1.0 + np.exp(-raw + 2.0))


class HawkesWinPredictor:
    """
    Profile-specific 2D Hawkes process for lead win-propensity scoring.
    Simplified implementation of Algorithm 1 from the paper.
    """

    def __init__(self, max_iter: int = 30, lr: float = 0.01):
        self.max_iter = max_iter
        self.lr = lr
        # Parameters to learn
        self.mu0_1  = 0.1    # base interaction intensity scale
        self.a11    = 0.3    # self-exciting coefficient
        self.w11    = 0.1    # kernel decay for interactions
        self.mu0_2  = 0.05   # base win intensity scale
        self.a21    = 0.5    # interaction -> win exciting coefficient
        self.w21    = 0.15   # kernel decay for win
        self.theta1 = None   # profile weights for interaction intensity
        self.theta2 = None   # profile weights for win intensity
        self.n_features = None

    def _profile_mu(self, theta: np.ndarray, mu0: float, x: np.ndarray) -> float:
        return mu0 * logistic(theta, x)

    def fit(self, interaction_sequences: list, profiles: np.ndarray,
            win_flags: list, T: float = 90.0):
        """
        Train the model using alternating optimization (Algorithm 1 from paper).

        Args:
            interaction_sequences: list of lists, each is sorted interaction timestamps for a lead
            profiles: np.ndarray of shape (n_leads, n_features) - normalized profile features
            win_flags: list of 0/1 indicating if lead was won
            T: observation window length in days
        """
        m = len(interaction_sequences)
        self.n_features = profiles.shape[1]
        self.theta1 = np.zeros(self.n_features)
        self.theta2 = np.zeros(self.n_features)

        print(f"[HawkesModel] Training on {m} leads for {self.max_iter} iterations...")

        for iteration in range(self.max_iter):
            # ── Step 1: Update p_ii and p_ij (E-step for MM algorithm) ──
            p_ii_list = []
            p_ij_list = []

            for s, times in enumerate(interaction_sequences):
                ns = len(times)
                mu_s = self._profile_mu(self.theta1, self.mu0_1, profiles[s])
                p_ii = []
                p_ij = []

                for i in range(ns):
                    exciting = sum(
                        self.a11 * exponential_kernel(times[i] - times[j], self.w11)
                        for j in range(i)
                    )
                    denom = mu_s + exciting + 1e-10
                    p_ii.append(mu_s / denom)
                    p_ij.append(exciting / denom)

                p_ii_list.append(p_ii)
                p_ij_list.append(p_ij)

            # ── Step 2: Update mu0_1, a11, w11 in closed form (Eq. 4,5,6) ──
            numerator_mu = 0.0
            denominator_mu = 0.0
            numerator_a = 0.0
            denominator_a = 0.0
            numerator_w = 0.0
            denominator_w = 0.0

            for s, times in enumerate(interaction_sequences):
                ns = len(times)
                h_s = logistic(self.theta1, profiles[s])
                numerator_mu += sum(p_ii_list[s]) / (h_s + 1e-10)
                denominator_mu += T

                for i in range(1, ns):
                    for j in range(i):
                        pij = (self.a11 * exponential_kernel(times[i] - times[j], self.w11)
                               / (self._profile_mu(self.theta1, self.mu0_1, profiles[s])
                                  + sum(self.a11 * exponential_kernel(times[i] - times[k], self.w11)
                                        for k in range(i)) + 1e-10))
                        numerator_a += pij
                        dt = times[i] - times[j]
                        numerator_w += pij
                        denominator_w += dt * pij

                for j in range(ns):
                    denominator_a += kernel_integral(T - times[j], self.w11)

            self.mu0_1 = max(numerator_mu / (denominator_mu + 1e-10), 1e-5)
            self.a11   = max(numerator_a  / (denominator_a  + 1e-10), 1e-5)
            self.w11   = max(numerator_w  / (denominator_w  + 1e-10), 1e-5)

            # ── Step 3: Gradient descent on theta1 (Eq. 8, 9, 10) ──
            grad1 = np.zeros(self.n_features)
            for s, times in enumerate(interaction_sequences):
                ns = len(times)
                h_s = logistic(self.theta1, profiles[s])
                mu_s = self.mu0_1 * h_s
                C = sum(
                    sum(self.a11 * exponential_kernel(times[i] - times[j], self.w11)
                        for j in range(i))
                    for i in range(ns)
                ) / max(ns, 1)
                factor = (ns / (mu_s + C + 1e-10) - T) * mu_s * (1 - h_s)
                grad1 += factor * profiles[s]

            self.theta1 -= self.lr * (-grad1)  # gradient ascent on log-likelihood

            # ── Step 4: Update outcome process parameters (L2) ──
            # Simplified: use logistic regression signal from win flags
            won_indices = [s for s, w in enumerate(win_flags) if w == 1]
            lost_indices = [s for s, w in enumerate(win_flags) if w == 0]

            if won_indices:
                # a21: average exciting effect on won leads
                exciting_won = []
                for s in won_indices:
                    times = interaction_sequences[s]
                    exc = sum(self.a21 * exponential_kernel(T - tj, self.w21) for tj in times)
                    exciting_won.append(exc)
                avg_exc_won = np.mean(exciting_won) if exciting_won else 0.1

                # Increase a21 slightly if it's informative
                self.a21 = min(self.a21 * 1.02, 2.0)

            # Gradient on theta2
            grad2 = np.zeros(self.n_features)
            for s, w in enumerate(win_flags):
                h_s = logistic(self.theta2, profiles[s])
                mu_s = self.mu0_2 * h_s
                target = float(w)
                error = target - (mu_s * 14.0)  # 14-day window
                grad2 += error * profiles[s] * h_s * (1 - h_s)

            self.theta2 += self.lr * grad2

            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:3d} | mu0_1={self.mu0_1:.4f} a11={self.a11:.4f} "
                      f"w11={self.w11:.4f} a21={self.a21:.4f}")

        print("[HawkesModel] Training complete.")

    def predict_proba(self, interaction_sequences: list, profiles: np.ndarray,
                      T: float = 90.0, window: float = 14.0) -> np.ndarray:
        """Return win propensity scores in [0,1] for each lead."""
        scores = []
        for s, times in enumerate(interaction_sequences):
            mu2_s = self._profile_mu(self.theta2, self.mu0_2, profiles[s])
            score = win_propensity_score(times, T, mu2_s, self.a21, self.w21, window)
            scores.append(score)
        return np.array(scores)