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
                # Si el brazo no ha sido seleccionado, lo priorizamos
                return a, 1  

            # Calcular τ(k_a) = (1 + alpha) ^ k_a
            tau_k_a = math.ceil((1 + self.alpha) ** self.epochs[a])
            ucb_values[a] = self.values[a] + np.sqrt(((1 + self.alpha) * np.log(math.e * self.t / tau_k_a)) / (2 * tau_k_a))

        # Seleccionar el brazo con el mayor valor UCB2
        arm = np.argmax(ucb_values)

        # Calcular el intervalo de ejecución
        tau_k_a = math.ceil((1 + self.alpha) ** self.epochs[arm])  # τ(k_a)
        tau_k_a_1 = math.ceil((1 + self.alpha) ** (self.epochs[arm] + 1))  # τ(k_a+1)
        intervalo_temporal = tau_k_a_1 - tau_k_a  # Diferencia de épocas

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

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.epochs = np.zeros(self.k, dtype=int)  # Reiniciar épocas
        self.t = 0  # Reiniciar el contador de tiempo
