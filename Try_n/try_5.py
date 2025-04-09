import numpy as np
from scipy.special import gamma

# Inicialización
num_hawks = 30  # Número de halcones
max_iterations = 1000  # Número máximo de iteraciones
beta = 1.5  # Parámetro para la función de Lévy
LB = np.array([65, 90, 40, 60, 20])  # Límite inferior de las valoraciones convertido a array
UB = np.array([85, 95, 60, 80, 30])  # Límite superior de las valoraciones convertido a array
cost_min = np.array([160, 300, 40, 100, 10])  # Costos mínimos convertidos a array
cost_max = np.array([200, 350, 80, 120, 20])  # Costos máximos convertidos a array
max_ads = np.array([15, 10, 25, 4, 30])  # Cantidad máxima de anuncios por tipo convertida a array

# Restricciones de costos
max_cost_tv = 3800
max_cost_dr = 2800
max_cost_drr = 3500

# Número de clientes estimados
num_customers = np.array([1000, 2000, 1500, 2500, 300])

# Función de costos lineales basada en la valoración
def cost_function(value, index):
    return cost_min[index] + (value - LB[index]) * (cost_max[index] - cost_min[index]) / (UB[index] - LB[index])

# Función de calidad total
def quality_function(values):
    return np.sum(values)

# Restricciones
def constraints(values):
    costs = np.array([cost_function(values[i], i) for i in range(len(values))])
    total_cost_tv = np.sum(costs[0:2] * max_ads[0:2])
    total_cost_dr = np.sum(costs[2:4] * max_ads[2:4])
    total_cost_drr = costs[2] * max_ads[2] + costs[4] * max_ads[4]
    total_cost = np.sum(costs * max_ads)

    if total_cost_tv > max_cost_tv or total_cost_dr > max_cost_dr or total_cost_drr > max_cost_drr or total_cost > 3800:
        return False
    return True

# Inicialización de los halcones
hawks = np.random.uniform(low=LB, high=UB, size=(num_hawks, len(LB)))
best_hawk = hawks[0]  # Asignación inicial de best_hawk
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
        r1, r2, r3, r4, r5 = np.random.rand(5)
        q = np.random.rand()

        # Fase de exploración
        if q >= 0.5:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        else:
            hawks[i] = (best_hawk - hawks.mean(axis=0)) - r3 * (LB + r4 * (UB - LB))

        # Actualización de la energía
        E0 = 2 * r1 - 1  # Energía inicial
        E = 2 * E0 * (1 - t / max_iterations)  # Energía actual

        # Fase de explotación
        if np.abs(E) >= 1:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
        else:
            if r1 >= 0.5 and np.abs(E) >= 0.5:
                J = 2 * (1 - r5)
                hawks[i] = best_hawk - E * np.abs(J * best_hawk - hawks[i])
            elif r1 >= 0.5 and np.abs(E) < 0.5:
                hawks[i] = best_hawk - E * np.abs(best_hawk - hawks[i])
            elif r1 < 0.5 and np.abs(E) >= 0.5:
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
