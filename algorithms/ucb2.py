"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo ucb2 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from algorithms.algorithm import Algorithm
import numpy as np
import math


class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        super().__init__(k)
        self.alpha = alpha
        self.t = 0
        self.ka = np.zeros(k, dtype=int) # Contador de épocas por brazo
        self.tau = np.zeros(k, dtype=int) # Tamaño de época por brazo

    def select_arm(self) -> int:
        self.t += 1

        for i in range(self.k):
            if self.counts[i] == 0: # Explorar cada brazo al menos una vez
                self.tau[i] = 1
                return i

        # Cálculo del valor UCB2 para cada brazo, según la fórmula de las diapositivas
        ucb_values = self.values + np.sqrt((1 + self.alpha) * np.log(math.e * self.t / self.tau) / (2 * self.tau))
        arm = np.argmax(ucb_values) # Elegimos el brazo  con mayor valor UCB

        # Actualizamos la época si ya hemos elegido este brazo `tau` veces
        if self.counts[arm] >= self.tau[arm]:
            self.ka[arm] += 1
            self.tau[arm] = math.ceil((1 + self.alpha) ** self.ka[arm])

        return arm
