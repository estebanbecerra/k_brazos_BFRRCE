from algorithms.algorithm import Algorithm
import numpy as np
import math

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2 con k brazos y un parámetro alpha para el balance entre exploración y explotación.
        
        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste (0 < alpha < 1).
        """
        assert 0 < alpha < 1, "El parámetro alpha debe ser mayor que 0 y menor que 1."
        super().__init__(k)
        self.alpha = alpha
        self.t = 0  # Contador de tiempo global
        self.ka = np.zeros(k, dtype=int)  # Número de épocas para cada brazo
        self.tau = np.ones(k, dtype=int)  # Tamaño de la época para cada brazo (inicialmente 1 para evitar problemas numéricos)

    def select_arm(self) -> tuple:
        """
        Selecciona el brazo a ejecutar de acuerdo a la política UCB2.
        
        Devuelve una tupla:
          (brazo seleccionado, intervalo de ejecución)
        
        El intervalo de ejecución es la cantidad de veces que se debe ejecutar ese brazo en la época actual.
        """
        self.t += 1  # Incrementa el contador de tiempo global
        
        # Explorar cada brazo al menos una vez
        for i in range(self.k):
            if self.counts[i] == 0:
                return i, 1  # Se devuelve el brazo y un intervalo de 1

        # Calcular el valor UCB2 para cada brazo:
        # ucb(a) = Q(a) + sqrt( ((1+alpha) * ln(e*t/τ(k_a)))/(2*τ(k_a)) )
        ucb_values = self.values + np.sqrt(
            (1 + self.alpha) * np.log(math.e * self.t / np.maximum(self.tau, 1)) / (2 * np.maximum(self.tau, 1))
        )
        ucb_values = np.nan_to_num(ucb_values, nan=-np.inf)  # Evitar NaNs

        arm = np.argmax(ucb_values)  # Selecciona el brazo con mayor UCB2

        # Calcular el intervalo de ejecución:
        tau_k_a = math.ceil((1 + self.alpha) ** self.ka[arm])
        tau_k_a_1 = math.ceil((1 + self.alpha) ** (self.ka[arm] + 1))
        intervalo_temporal = tau_k_a_1 - tau_k_a

        return arm, intervalo_temporal

    def update(self, chosen_arm: int, reward: float, update_epoch: bool):
        """
        Actualiza la estimación de la recompensa y, opcionalmente, el número de épocas (ka) y el tamaño de la época (tau)
        para el brazo seleccionado.
        
        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        :param update_epoch: Si es True, se actualiza la época para el brazo.
        """
        # Actualización estándar del valor y el contador
        super().update(chosen_arm, reward)
        
        # Si se requiere actualizar la época para el brazo seleccionado, se incrementa ka y se recalcula tau
        if update_epoch:
            self.ka[chosen_arm] += 1
            self.tau[chosen_arm] = min(math.ceil((1 + self.alpha) ** self.ka[chosen_arm]), 10**6)

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.ka = np.zeros(self.k, dtype=int)
        self.tau = np.ones(self.k, dtype=int)
        self.t = 0
