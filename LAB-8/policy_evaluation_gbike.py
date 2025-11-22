import numpy as np
import math


def policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam):
    m, n = policy.shape
    max_n = max(13, 15, 10)
    nn = np.arange(max_n)
    P1 = np.exp(-Lamda[0]) * (Lamda[0] ** nn) / np.vectorize(math.factorial)(nn)
    P2 = np.exp(-Lamda[1]) * (Lamda[1] ** nn) / np.vectorize(math.factorial)(nn)
    P3 = np.exp(-lamda[0]) * (lamda[0] ** nn) / np.vectorize(math.factorial)(nn)
    P4 = np.exp(-lamda[1]) * (lamda[1] ** nn) / np.vectorize(math.factorial)(nn)
    V = np.zeros((m, n))
    delta = 10
    theta = 0.1

    while delta > theta:
        v = V.copy()
        for i in range(m):
            for j in range(n):
                s1 = i - 1
                s2 = j - 1
                Vs_ = 0
                a = policy[i, j]
                R = -abs(a) * t
                s1_ = s1 - a
                s2_ = s2 + a
                for n1 in range(13):
                    for n2 in range(15):
                        s1__ = s1_ - min(n1, s1_)
                        s2__ = s2_ - min(n2, s2_)
                        for n3 in range(13):
                            for n4 in range(10):
                                s1___ = s1__ + min(n3, 20 - s1__)
                                s2___ = s2__ + min(n4, 20 - s2__)

                                if 0 <= s1___ < m and 0 <= s2___ < n:
                                    R += (
                                        P1[n1]
                                        * P2[n2]
                                        * P3[n3]
                                        * P4[n4]
                                        * (min(n1, s1_) + min(n2, s2_))
                                    ) * r
                                    Vs_ += (
                                        P1[n1]
                                        * P2[n2]
                                        * P3[n3]
                                        * P4[n4]
                                        * V[int(s1___), int(s2___)]
                                    )
                V[i, j] = R + (gam * Vs_)
        delta = np.max(np.abs(v - V))
    return V