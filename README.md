# Un estudio comparativo entre distintos algoritmos de las familias epsilon-greedy, UCB y Ascenso de Gradiente

## Información
- **Alumnos:** Becerra Fernández, Esteban; Cruzado Esteban, Carlos; Ruzhytska Ruzhytska, Anastasiya
- **Asignatura:** Extensiones de Machine Learning
-  **Curso:** 2024/2025
-  **Grupo:** BCR

## Descripción
El objetivo de este trabajo es implementar una serie de algoritmos pertenecientes a las familias epsilon-greedy, UCB y Ascenso de Gradiente y poder comprobar su desempeño en un problema de exploración-explotación como lo es el problema del Bandido de K-brazos. El estudio se llevará a cabo variando los hiperparámetros propios de cada algoritmo así como la distribución de las recompensas del bandido, característica que también será implementada. A saber, distribución Normal, Bernoulli y Binomial. El análisis de rendimiento será contrastado con una serie de métricas y gráficas que permitan la visualización intuitiva de los resultados obtenidos.

## Estructura
El repositorio consta de dos carpetas: 
-  "src" -> Aquí encontramos los ficheros relativos al código del proyecto. Dividido en las siguientes subcarpetas:
      - "algorithms" -> Aquí se encuentran los algoritmos implementados.
      - "arms" -> Aquí se encuentran las configuraciones de los bandidos para cada tipo de distribución.
      - "plotting" -> Aquí se encuentran los ficheros relativos a la visualización gráfica de las características de los algoritmos y bandidos.
- "docs" -> Aquí se encuentra el fichero pdf relativo a la documentación del proyecto.
- "README.MD" -> Fichero actual, explicación de la organización, estructura e instrucciones de uso del proyecto.
- "notebook1.ipynb" -> Breve introducción del problema
- "notebook2.ipynb" -> Estudio de la familia epsilon-greedy
- "notebook3.ipynb" -> Estudio de la familia UCB
- "notebook4.ipynb" -> Estudio de la familia de Ascenso de Gradiente

## Instalación y Uso
Para poder observar los experimentos realizados así como el estudio llevado a cabo se recomienda abrir en Colab el documento "main.ipynb", ejecutarlo para importar los archivos necesarios, y desde el mismo navegar por los otros notebooks implementados, titulados "Notebook1.ipynb", "Notebook2.ipynb", "Notebook3.ipynb", "Notebook4.ipynb".

## Tecnologías Utilizadas
  - Lenguaje de programación: Python
  - Frameworks y librerías: NumPy, MAtplotlib, ABC, Typing, Math, Random
