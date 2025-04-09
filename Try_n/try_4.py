'''
Intento luego de la comprencion del paper HHO
'''


import numpy as np
from scipy.special import gamma

'''
Explicacion del codigo: Inicializacion
Inicializamos los parámetros del algoritmo, incluyendo el número de halcones,
el número máximo de iteraciones, el parámetro beta para la función de Lévy,
y los límites y costos de los anuncios.
'''
# Inicialización
num_hawks = 30  # Número de halcones
max_iterations = 1000  # Número máximo de iteraciones
beta = 1.5  # Parámetro para la función de Lévy
LB = [65, 90, 40, 60, 20]  # Límite inferior de las valoraciones
UB = [85, 95, 60, 80, 30]  # Límite superior de las valoraciones
cost_min = [160, 300, 40, 100, 10]  # Costos mínimos
cost_max = [200, 350, 80, 120, 20]  # Costos máximos
max_ads = [15, 10, 25, 4, 30]  # Cantidad máxima de anuncios por tipo

# Restricciones de costos
max_cost_tv = 3800
max_cost_dr = 2800
max_cost_drr = 3500

# Número de clientes estimados
num_customers = [1000, 2000, 1500, 2500, 300]

'''
Explicacion del codigo: Funciones Auxiliares
Definimos funciones auxiliares para calcular los costos basados en la valoración,
la calidad total de las valoraciones, y para verificar si una solución cumple
con las restricciones de costo.
'''
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

'''
Explicacion del codigo: Inicializacion de los Halcones
Generamos posiciones iniciales aleatorias para los halcones dentro de los límites
especificados. También inicializamos la mejor posición del halcón y la mejor calidad.
'''
# Inicialización de los halcones
hawks = np.random.uniform(low=LB, high=UB, size=(num_hawks, len(LB)))
best_hawk = hawks[0]
best_quality = -np.inf

'''
Explicacion del codigo: Función de Lévy
Definimos la función de Lévy para calcular los pasos de vuelo de Lévy, 
simulando movimientos aleatorios en zigzag.
'''
# Función de Lévy
def levy_flight(beta, dim):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

'''
Explicacion del codigo: Bucle Principal del Algoritmo HHO
Actualizamos las posiciones de los halcones según las fases de exploración y explotación,
basadas en las ecuaciones del HHO. Evaluamos las soluciones y mantenemos la mejor solución
encontrada.
'''
# Bucle principal del algoritmo HHO
for t in range(max_iterations):
    for i in range(num_hawks):
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        r4 = np.random.rand()
        r5 = np.random.rand()
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
