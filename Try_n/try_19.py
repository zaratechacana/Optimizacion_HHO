import numpy as np
import random
import math

# Definimos los dominios de las variables
domains = {
    'x_1': list(range(16)),  # x_1 <= 15
    'x_2': list(range(11)),  # x_2 <= 10
    'x_3': list(range(26)),  # x_3 <= 25
    'x_4': list(range(5)),   # x_4 <= 4
    'x_5': list(range(31))   # x_5 <= 30
}

# Definimos restricciones basadas en las restricciones lineales
constraints = {
    ('x_1', 'x_2'): lambda x1, x2: 160*x1 + 300*x2 <= 3800,
    ('x_3', 'x_4'): lambda x3, x4: 40*x3 + 100*x4 <= 2800,
    ('x_3', 'x_5'): lambda x3, x5: 40*x3 + 10*x5 <= 3500
}

# Modificamos la función revise para que utilice las restricciones lineales
def revise(x, y):
    revised = False
    to_remove = []
    for x_value in domains[x]:
        satisfies = any(constraints[(x, y)](x_value, y_value) for y_value in domains[y] if (x, y) in constraints)
        if not satisfies:
            to_remove.append(x_value)
    if to_remove:
        domains[x] = [value for value in domains[x] if value not in to_remove]
        revised = True
    return revised


# Algoritmo AC3 para garantizar la consistencia
def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        revised = revise(x, y)
        if revised:
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x]
            queue.extend(neighbors)


# Definimos los arcos basados en las dependencias entre variables
arcs = [('x_1', 'x_2'), ('x_3', 'x_4'), ('x_3', 'x_5')]

# Ejecutamos el algoritmo AC3
ac3(arcs)
print("Dominios después de AC3:")
for key, value in domains.items():
    print(f"{key}: {value}\n")

def verificar_restricciones(x):
    x1, x2, x3, x4, x5 = x

    if not (0 <= x1 <= 15) or not (0 <= x2 <= 10) or not (0 <= x3 <= 25) or not (0 <= x4 <= 4) or not (0 <= x5 <= 30):
        return False
    if not (160 * x1 + 300 * x2 <= 3800):
        return False
    if not (40 * x3 + 100 * x4 <= 2800):
        return False
    if not (40 * x3 + 10 * x5 <= 3500):
        return False

    return True

def fit_maxi(solution, values):
    return sum(s * v for s, v in zip(solution, values))

def fit_min(solution, costs):
    return sum(s * c for s, c in zip(solution, costs))

def fit(x, w_p, w_q, e_p_xbest, e_q_xbest, C):
    z1 = 65*x[0] + 85*x[1] + 40*x[2] + 60*x[3] + 30*x[4]
    z2 = 160*x[0] + 300*x[1] + 40*x[2] + 100*x[3] + 10*x[4]
    j_new = ((z1 / e_p_xbest) * w_p) + (((C - z2) / (C - e_q_xbest)) * w_q)
    return j_new

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter, w_p, w_q, e_pq_xbest, e_q_xbest, C):
    Rabbit_Location = np.array([15, 2, 25, 4, 30])  # Mejor posición inicial basada en el conocimiento previo
    Rabbit_Energy = -np.inf  # Inicializar como el peor caso posible para maximizar

    # Inicialización de posiciones de los halcones
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))

    # Bucle principal
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Calcular la aptitud
            fitness = objf(X[i, :], w_p, w_q, e_pq_xbest, e_q_xbest, C)
            # Actualizar la ubicación del conejo
            if fitness > Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        # Calcular la energía de escape
        E1 = 2 * (1 - (t / Max_iter))

        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            r = random.uniform(0, 1)
            Jump_strength = 2 * (1 - random.uniform(0, 1))

            if abs(Escaping_Energy) < 1:  # Explotación
                if r >= 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])
                else:
                    X[i, :] = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])
            else:  # Exploración
                X[i, :] = np.random.uniform(lb, ub, dim)

            # Aplicar límites
            X[i, :] = np.clip(X[i, :], lb, ub)

            # Actualizar fitness tras el movimiento
            new_fitness = objf(X[i, :], w_p, w_q, e_pq_xbest, e_q_xbest, C)
            if new_fitness > fitness:
                fitness = new_fitness
                Rabbit_Location = X[i, :].copy()
                Rabbit_Energy = new_fitness

        if t % 100 == 0:
            print(f'At iteration {t}, the best fitness is {Rabbit_Energy}')

    return Rabbit_Location, Rabbit_Energy

# Parámetros de la función objetivo
w_p = 1 # Peso para el costo
w_q = 0 # Peso para la calidad
e_p_xbest = 3175  # Valor óptimo de Z_1 obtenido max
e_q_xbest = 0  # Valor óptimo de Z_2 obtenido min
C = 10000  # Cota superior para Z_2 como calculado previamente

# Parámetros del HHO
lb = np.zeros(5)
ub = np.array([15, 10, 25, 4, 30])
dim = 5
SearchAgents_no = 1
Max_iter = 100

best_position, best_quality = HHO(fit, lb, ub, dim, SearchAgents_no, Max_iter, w_p, w_q, e_p_xbest, e_q_xbest, C)
print("Best Position:", np.round(best_position).astype(int))
print("Best Quality:", best_quality)