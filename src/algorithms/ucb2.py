from algorithms.algorithm import Algorithm
import numpy as np
import math

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2 con k brazos y un parámetro de ajuste alpha.
        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para el balance entre exploración y explotación.
        """
        assert 0 < alpha < 1, "El parámetro alpha debe ser mayor que 0 y menor que 1."

        super().__init__(k)
        self.alpha = alpha
        self.t = 0  # Contador de tiempo global
        self.ka = np.zeros(k, dtype=int)  # Número de épocas por brazo
        self.tau = np.ones(k, dtype=int)  # Tamaño de la época por brazo (inicialmente en 1)

    def select_arm(self) -> tuple:
        """
        Selecciona el brazo con el mayor valor UCB2 y devuelve el intervalo de ejecución correspondiente.
        :return: Índice del brazo seleccionado y el intervalo temporal en el que se ejecutará.
        """
        self.t += 1  # Incrementamos el tiempo global

        for i in range(self.k):
            if self.counts[i] == 0:  # Explorar cada brazo al menos una vez
                return i, 1  # Devuelve el brazo y un intervalo de 1

        # Cálculo del valor UCB2 para cada brazo
        ucb_values = self.values + np.sqrt(
            (1 + self.alpha) * np.log(math.e * self.t / np.maximum(self.tau, 1)) / (2 * np.maximum(self.tau, 1))
        )

        # Evitar posibles NaNs
        ucb_values = np.nan_to_num(ucb_values, nan=-np.inf)

        # Seleccionamos el brazo con el índice UCB más alto
        arm = np.argmax(ucb_values)

        # Calculamos τ(k_a) y τ(k_a+1) para determinar el intervalo de ejecución
        tau_k_a = math.ceil((1 + self.alpha) ** self.ka[arm])
        tau_k_a_1 = math.ceil((1 + self.alpha) ** (self.ka[arm] + 1))
        intervalo_temporal = tau_k_a_1 - tau_k_a

        return arm, intervalo_temporal

    def update(self, chosen_arm: int, reward: float, update_epoch: bool):
        """
        Actualiza el valor del brazo seleccionado y gestiona las épocas.
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        :param update_epoch: Indica si debe actualizar la época del brazo.
        """
        super().update(chosen_arm, reward)

        if update_epoch:
            self.ka[chosen_arm] += 1  # Incrementamos la cantidad de épocas del brazo
            self.tau[chosen_arm] = min(math.ceil((1 + self.alpha) ** self.ka[chosen_arm]), 10**6)  # Evita overflow

    def reset(self):
        """
        Reinicia el estado del algoritmo UCB2.
        """
        super().reset()
        self.ka = np.zeros(self.k, dtype=int)  # Reinicia las épocas de cada brazo
        self.tau = np.ones(self.k, dtype=int)  # Reinicia los tamaños de época
        self.t = 0  # Reinicia el contador de tiempo
