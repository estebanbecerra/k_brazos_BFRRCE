import numpy as np
import math

class UCB2:
    def __init__(self, n_arms, alpha=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)  # Número de veces que se ha seleccionado cada brazo
        self.values = np.zeros(n_arms)  # Recompensa promedio de cada brazo
        self.epochs = np.zeros(n_arms, dtype=int)  # Épocas por brazo
        self.tau = np.ones(n_arms, dtype=int)  # Duración de cada época

    def select_arm(self):
        """Selecciona el brazo con el índice UCB2 más alto."""
        # Si algún brazo no ha sido probado, seleccionarlo primero
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Calcular UCB2 para cada brazo
        ucb_values = self.values + np.sqrt((1 + self.alpha) * np.log(math.e * sum(self.counts) / self.tau) / (2 * self.tau))
        
        return np.argmax(ucb_values)  # Seleccionamos el brazo con el mayor índice

    def update(self, chosen_arm, reward):
        """Actualiza las estadísticas del brazo seleccionado."""
        self.counts[chosen_arm] += 1
        self.epochs[chosen_arm] += 1  # Incrementar el número de épocas del brazo
        
        # Actualizar estimación de la recompensa media
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

        # Actualizar τ (duración de la siguiente época)
        self.tau[chosen_arm] = max(math.ceil((1 + self.alpha) ** self.epochs[chosen_arm]), 1)
