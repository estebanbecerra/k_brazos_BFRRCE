from algorithms.algorithm import Algorithm
import numpy as np
import math

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2 con k brazos y un parámetro alpha para el balance entre exploración y explotación.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste (0 < alpha < 1), controla la frecuencia de exploración.
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar en el rango (0,1)."
        super().__init__(k)  # Llama al constructor de Algorithm
        self.alpha = alpha
        self.epochs = np.zeros(k, dtype=int)  # Número de épocas por brazo
        self.tau = np.ones(k, dtype=int)  # Tamaño de la época por brazo
        self.MAX_TAU = 10_000  # 🔹 Límite máximo de tau para evitar overflow

    def select_arm(self) -> int:
        """
        Selecciona el brazo con el índice UCB2 más alto.
        :return: Índice del brazo seleccionado.
        """
        # Explorar cada brazo al menos una vez
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
        
        # Calcular UCB2 para cada brazo según la ecuación teórica
        total_count = sum(self.counts)
        ucb_values = self.values + np.sqrt(
            (1 + self.alpha) * np.log(math.e * total_count / np.maximum(self.tau, 1)) / (2 * np.maximum(self.tau, 1))
        )

        return np.argmax(ucb_values)  # Seleccionar el brazo con el mayor índice UCB2

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la recompensa promedio del brazo seleccionado y su época.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        super().update(chosen_arm, reward)  # Usa la actualización de `Algorithm`

        # Actualizar número de épocas del brazo
        self.epochs[chosen_arm] += 1

        # Actualizar τ (duración de la siguiente época) con un límite máximo
        self.tau[chosen_arm] = min(math.ceil((1 + self.alpha) ** self.epochs[chosen_arm]), self.MAX_TAU)

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()  # Llama al reset de Algorithm
        self.epochs = np.zeros(self.k, dtype=int)  # Reiniciar épocas
        self.tau = np.ones(self.k, dtype=int)  # Reiniciar tamaño de época
