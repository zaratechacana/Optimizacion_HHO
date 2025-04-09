import math
import numpy as np
from scipy.special import gamma

# Definición de los dominios de las variables
domains = {
    'x1': list(range(16)),  # 0 a 15
    'x2': list(range(11)),  # 0 a 10
    'x3': list(range(26)),  # 0 a 25
    'x4': list(range(5)),   # 0 a 4
    'x5': list(range(31))   # 0 a 30
}

# Función de calidad
def quality_function(x1, x2, x3, x4, x5):
    quality_scores = np.array([85, 95, 60, 80, 30])  # Puntuaciones de calidad
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(quality_scores, quantities)

# Función de costo
def cost_function(x1, x2, x3, x4, x5):
    costs = np.array([200, 350, 80, 120, 20])  # Costos
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(costs, quantities)

# Función para verificar el cumplimiento del límite de costo
def cost_constraint(x1, x2, x3, x4, x5):
    return cost_function(x1, x2, x3, x4, x5) <= 3800

# Restricciones, incluyendo las restricciones adicionales
def constraints(x1, x2, x3, x4, x5):
    if (200 * x1 + 350 * x2 > 3800 or
        80 * x3 + 120 * x4 > 2800 or
        80 * x3 + 20 * x5 > 3500 or
        cost_function(x1, x2, x3, x4, x5) > 3800):
        return False
    return True

# Función de Lévy
def levy_flight(beta, dim):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

# Configuración inicial para la simulación
num_hawks = 10  # Número de halcones en la simulación
max_iterations = 50  # Número máximo de iteraciones
beta = 1.5  # Parámetro para la función de Lévy
LB = np.array([65, 90, 40, 60, 20])  # Límites inferiores para las variables
UB = np.array([85, 95, 60, 80, 30])  # Límites superiores para las variables

# Inicialización de los halcones
hawks = np.random.uniform(low=LB, high=UB, size=(num_hawks, len(LB)))
hawks = np.round(hawks).astype(int)
best_hawk = np.copy(hawks[0])
best_quality = quality_function(*best_hawk)

# Encontrar la mejor solución inicial
for i in range(num_hawks):
    if constraints(*hawks[i]):
        current_quality = quality_function(*hawks[i])
        if current_quality > best_quality:
            best_quality = current_quality
            best_hawk = hawks[i]

# Si no se encontró ninguna solución válida, inicializar best_hawk con el primer halcón válido
if best_hawk is None:
    best_hawk = hawks[0]
    best_quality = quality_function(*best_hawk)

# Bucle principal del algoritmo HHO
for t in range(max_iterations):
    for i in range(num_hawks):
        r1, r2, r3, r4, r5 = np.random.rand(5)
        q = np.random.rand()

        # Fase de exploración
        if q >= 0.5:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        elif q < 0.5:
            hawks[i] = (best_hawk - hawks.mean(axis=0)) - r3 * (LB + r4 * (UB - LB))

        # Asegurarse de que los valores estén dentro de los límites
        hawks[i] = np.clip(hawks[i], LB, UB)
        hawks[i] = np.round(hawks[i]).astype(int)

        # Actualización de la energía
        E0 = 2 * r1 - 1  # Energía inicial
        E = 2 * E0 * (1 - t / max_iterations)  # Energía actual

        # Fase de explotación
        if np.abs(E) >= 1:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        elif np.abs(E) < 1:
            if r1 >= 0.5 and np.abs(E) >= 0.5:
                J = 2 * (1 - r5)
                hawks[i] = best_hawk - E * np.abs(J * best_hawk - hawks[i])
            elif r1 >= 0.5 and np.abs(E) < 0.5:
                hawks[i] = best_hawk - E * np.abs(best_hawk - hawks[i])
            elif r1 < 0.5 and np.abs(E) >= 0.5:
                J = 2 * (1 - r5)
                Y = best_hawk - E * np.abs(J * best_hawk - hawks[i])
                Z = Y + np.random.rand(len(LB)) * levy_flight(beta, len(LB))
                if quality_function(*Y) > quality_function(*hawks[i]) and constraints(*Y):
                    hawks[i] = Y
                elif quality_function(*Z) > quality_function(*hawks[i]) and constraints(*Z):
                    hawks[i] = Z
            elif r1 < 0.5 and np.abs(E) < 0.5:
                J = 2 * (1 - r5)
                Y = best_hawk - E * np.abs(J * best_hawk - hawks.mean(axis=0))
                Z = Y + np.random.rand(len(LB)) * levy_flight(beta, len(LB))
                if quality_function(*Y) > quality_function(*hawks[i]) and constraints(*Y):
                    hawks[i] = Y
                elif quality_function(*Z) > quality_function(*hawks[i]) and constraints(*Z):
                    hawks[i] = Z

        # Asegurarse de que los valores estén dentro de los límites después de la fase de explotación
        hawks[i] = np.clip(hawks[i], LB, UB)
        hawks[i] = np.round(hawks[i]).astype(int)

    # Evaluar la mejor solución
    for i in range(num_hawks):
        if constraints(*hawks[i]):
            current_quality = quality_function(*hawks[i])
            if current_quality > best_quality:
                best_quality = current_quality
                best_hawk = hawks[i]

# Resultados
print(f"Mejor calidad de anuncios: {best_quality}")
print(f"Configuración de anuncios: {best_hawk}")

# Mostrar los resultados finales
q1, q2, q3, q4, q5 = 85, 95, 60, 80, 30  # Valores de calidad correspondientes
print(f"x1 = {best_hawk[0]}")
print(f"x2 = {best_hawk[1]}")
print(f"x3 = {best_hawk[2]}")
print(f"x4 = {best_hawk[3]}")
print(f"x5 = {best_hawk[4]}")
print(f"q1 = {q1}")
print(f"q2 = {q2}")
print(f"q3 = {q3}")
print(f"q4 = {q4}")
print(f"q5 = {q5}")
print(f"Calidad Total = {best_quality}")
