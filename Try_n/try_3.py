"3er intento zarate en funcion del  video de youtube"
#https://www.youtube.com/watch?v=XSttSi2vMm8&t=463s


import numpy as np
from scipy.special import gamma

# Inicialización
num_hawks = 30  # Número de halcones
max_iterations = 1000  # Número máximo de iteraciones
beta = 1.5  # Parámetro para la función de Lévy
LB = [65, 90, 40, 60, 20]  # Límite inferior de las valoraciones
UB = [85, 95, 60, 80, 30]  # Límite superior de las valoraciones
cost_min = [160, 300, 40, 100, 10]  # Costos mínimos
cost_max = [200, 350, 80, 120, 20]  # Costos máximos
max_ads = [15, 10, 25, 4, 30]  # Cantidad máxima de anuncios por tipo
max_cost_tv = 3800
max_cost_dr = 2800
max_cost_drr = 3500
num_customers = [1000, 2000, 1500, 2500, 300]

# Función de costos lineales basada en la valoración
def cost_function(value, index):
    return cost_min[index] + (value - LB[index]) * (cost_max[index] - cost_min[index]) / (UB[index] - LB[index])

# Función de calidad total
def quality_function(values):
    return sum(values)

# Restricciones
def constraints(values):
    costs = [cost_function(values[i], i) for i in range(len(values))]
    total_cost_tv = costs[0] * max_ads[0] + costs[1] * max_ads[1]
    total_cost_dr = costs[2] * max_ads[2] + costs[3] * max_ads[3]
    total_cost_drr = costs[2] * max_ads[2] + costs[4] * max_ads[4]
    total_cost = sum(costs[i] * max_ads[i] for i in range(len(costs)))

    if total_cost_tv > max_cost_tv or total_cost_dr > max_cost_dr or total_cost_drr > max_cost_drr or total_cost > 3800:
        return False
    return True

# Inicialización de los halcones
hawks = np.random.uniform(low=LB, high=UB, size=(num_hawks, len(LB)))
best_hawk = None
best_quality = -np.inf

# Función de Lévy
def levy_flight(beta, dim):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

# Bucle principal del algoritmo HHO
for t in range(max_iterations):
    for i in range(num_hawks):
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        r4 = np.random.rand()
        r5 = np.random.rand()
        q = np.random.rand()

        if q >= 0.5:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        else:
            hawks[i] = (best_hawk - hawks.mean(axis=0)) - r3 * (LB + r4 * (UB - LB))

        E0 = 2 * r1 - 1  # Energía inicial
        E = 2 * E0 * (1 - t / max_iterations)  # Energía actual

        if np.abs(E) >= 1:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        else:
            if r >= 0.5 and np.abs(E) >= 0.5:
                J = 2 * (1 - r5)
                hawks[i] = best_hawk - E * np.abs(J * best_hawk - hawks[i])
            elif r >= 0.5 and np.abs(E) < 0.5:
                hawks[i] = best_hawk - E * np.abs(best_hawk - hawks[i])
            elif r < 0.5 and np.abs(E) >= 0.5:
                J = 2 * (1 - r5)
                Y = best_hawk - E * np.abs(J * best_hawk - hawks[i])
                Z = Y + np.random.rand(len(LB)) * levy_flight(beta, len(LB))
                if quality_function(Y) > quality_function(hawks[i]) and constraints(Y):
                    hawks[i] = Y
                elif quality_function(Z) > quality_function(hawks[i]) and constraints(Z):
                    hawks[i] = Z
            else:
                J = 2 * (1 - r5)
                Y = best_hawk - E * np.abs(J * best_hawk - hawks.mean(axis=0))
                Z = Y + np.random.rand(len(LB)) * levy_flight(beta, len(LB))
                if quality_function(Y) > quality_function(hawks[i]) and constraints(Y):
                    hawks[i] = Y
                elif quality_function(Z) > quality_function(hawks[i]) and constraints(Z):
                    hawks[i] = Z

    # Evaluar la mejor solución
    for i in range(num_hawks):
        if quality_function(hawks[i]) > best_quality and constraints(hawks[i]):
            best_quality = quality_function(hawks[i])
            best_hawk = hawks[i]

# Resultados
print("Mejor calidad de anuncios:", best_quality)
print("Configuración de anuncios:", best_hawk)


