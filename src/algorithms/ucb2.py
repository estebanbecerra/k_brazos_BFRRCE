from algorithms.algorithm import Algorithm
import numpy as np
import math

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2 con k brazos y un parámetro alpha para el balance entre exploración y explotación.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste (0 < alpha < 1), controla la frecuencia de exploración.
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar en el rango (0,1)."
        super().__init__(k)
        self.alpha = alpha
        self.t = 0  # Contador de tiempo global
        self.epochs = np.zeros(k, dtype=int)  # Número de épocas para cada brazo
        self.tau = np.ones(k, dtype=int)  # Tamaño de la época por brazo
        self.MAX_TAU = 10**6  # Límite máximo para evitar overflow

    def select_arm(self) -> tuple:
        """
        Selecciona el brazo con el mayor índice UCB2.
        
        Devuelve:
          (índice del brazo seleccionado, intervalo de ejecución)

        El intervalo de ejecución es la cantidad de veces que se ejecutará el brazo en la época actual.
        """
        self.t += 1  # Incrementa el contador de tiempo global

        ucb_values = np.zeros(self.k)

        for a in range(self.k):
            if self.counts[a] == 0:
                return a, 1  # Se explora cada brazo al menos una vez

            # Calcular τ(k_a) = (1 + alpha) ^ epochs[a], asegurando que no crezca demasiado
            tau_k_a = min(math.ceil((1 + self.alpha) ** self.epochs[a]), self.MAX_TAU)
            ucb_values[a] = self.values[a] + np.sqrt(((1 + self.alpha) * np.log(max(math.e * self.t / tau_k_a, 1))) / (2 * tau_k_a))

        # Seleccionar el brazo con el mayor valor UCB2
        arm = np.argmax(ucb_values)

        # Calcular el intervalo de ejecución
        tau_k_a = min(math.ceil((1 + self.alpha) ** self.epochs[arm]), self.MAX_TAU)
        tau_k_a_1 = min(math.ceil((1 + self.alpha) ** (self.epochs[arm] + 1)), self.MAX_TAU)
        intervalo_temporal = max(tau_k_a_1 - tau_k_a, 1)  # Evitar intervalos de 0

        return arm, intervalo_temporal

    def update(self, chosen_arm: int, reward: float, update_epoch: bool):
        """
        Actualiza la estimación de la recompensa y, opcionalmente, el número de épocas para el brazo seleccionado.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        :param update_epoch: Si es True, se actualiza la época para el brazo.
        """
        # Actualización estándar del valor y el contador
        super().update(chosen_arm, reward)

        # Si se requiere actualizar la época para el brazo seleccionado
        if update_epoch:
            self.epochs[chosen_arm] += 1  # Incrementar la cantidad de épocas del brazo
            self.tau[chosen_arm] = min(math.ceil((1 + self.alpha) ** self.epochs[chosen_arm]), self.MAX_TAU)  # Evita τ(k_a) < 1

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.epochs = np.zeros(self.k, dtype=int)  # Reiniciar épocas
        self.tau = np.ones(self.k, dtype=int)  # Reiniciar tamaño de época
        self.t = 0  # Reiniciar el contador de tiempo
