import numpy as np
import random
import math

# Definición de los dominios de las variables
domains = {
    'x1': list(range(16)),  # 0 a 15
    'x2': list(range(11)),  # 0 a 10
    'x3': list(range(26)),  # 0 a 25
    'x4': list(range(5)),   # 0 a 4
    'x5': list(range(31))   # 0 a 30
}

# Pesos para cada objetivo
w_cost = 0.1  # Peso para el costo
w_quality = 0.9  # Peso para la calidad


# Función de costo
def cost_function(x1, x2, x3, x4, x5):
    costs = np.array([180, 325, 60, 110, 15])  # Costos promedio
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(costs, quantities)

# Función escalonada
def scalarized_objective(x1, x2, x3, x4, x5):
    total_quality = quality_function(x1, x2, x3, x4, x5)
    total_cost = cost_function(x1, x2, x3, x4, x5)
    return w_cost * total_cost - w_quality * total_quality  # Minimizar la función escalonada

epsilon = 5000  # Calidad mínima requerida

# Restricciones, incluyendo la calidad
constraints = {
    ('x1', 'x2'): lambda x1, x2: 160 * x1 + 300 * x2 <= 3800,
    ('x3', 'x4'): lambda x3, x4: 40 * x3 + 100 * x4 <= 2800,
    ('x3', 'x5'): lambda x3, x5: 40 * x3 + 10 * x5 <= 3500,
    'quality': lambda x1, x2, x3, x4, x5: quality_function(x1, x2, x3, x4, x5) >= epsilon
}

# Revisar y actualizar los dominios de acuerdo a todas las restricciones
def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [constraints[constraint] for constraint in constraints if constraint[0] == x and constraint[1] == y]
    for x_value in x_domain[:]:
        satisfies = False
        for y_value in y_domain:
            if all(constraint(x_value, y_value) for constraint in all_constraints):
                satisfies = True
                break
        if not satisfies:
            x_domain.remove(x_value)
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

arcs = [
    ('x1', 'x2'), ('x2', 'x1'), 
    ('x3', 'x4'), ('x4', 'x3'), 
    ('x3', 'x5'), ('x5', 'x3')
]

ac3(arcs)

print("Dominios después de AC3:")
for key, value in domains.items():
    print(f"{key}: {value}\n")

# Aquí necesitarás implementar el algoritmo de optimización que desees usar para encontrar la mejor solución.


# Función de calidad ajustada con los nuevos valores
def quality_function(x):
    quality_scores = np.array([75, 92.5, 50, 70, 25])
    return np.dot(quality_scores, x)

# Función de costo ajustada con los nuevos valores
def cost_function(x):
    costs = np.array([180, 325, 60, 110, 15])
    return np.dot(costs, x)

# Función objetivo para HHO
def objective_function(x):
    quality = quality_function(x)
    cost = cost_function(x)
    c1, c2, c3, c4, c5 = 180, 325, 60, 110, 15
    
    # Penalización si alguna restricción no se cumple
    if cost > 3800:
        return -float('inf')
    if c1 * x[0] + c2 * x[1] > 3800:
        return -float('inf')
    if c3 * x[2] + c4 * x[3] > 2800:
        return -float('inf')
    if c3 * x[2] + c5 * x[4] > 3500:
        return -float('inf')
    return quality

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

# Implementación de HHO adaptada para el problema específico
def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Inicialización de ubicación y energía del conejo con los valores óptimos conocidos
    Rabbit_Location = np.array([15, 2, 25, 4, 30])
    Rabbit_Energy = objective_function(Rabbit_Location)

    # Inicialización de posiciones de los halcones
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    
    # Bucle principal
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Aplicar límites
            X[i, :] = np.clip(X[i, :], lb, ub)
            
            # Calcular la aptitud
            fitness = objf(X[i, :])
            
            # Actualizar la ubicación del conejo
            if fitness > Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        # Calcular la energía de escape
        E1 = 2 * (1 - (t / Max_iter))
        
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) < 1:  # Fase de explotación
                r = random.uniform(0, 1)
                Jump_strength = 2 * (1 - random.uniform(0, 1))
                
                # Diferentes estrategias basadas en el valor de r y la energía de escape
                if r >= 0.5:
                    if abs(Escaping_Energy) >= 0.5:  # Soft besiege
                        X[i, :] = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    else:  # Hard besiege
                        X[i, :] = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                else:
                    if abs(Escaping_Energy) >= 0.5:  # Soft besiege with progressive rapid dives
                        X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                        X1 = np.clip(X1, lb, ub)
                        if objf(X1) > fitness:
                            X[i, :] = X1
                        else:
                            X2 = X1 + np.random.randn(dim) * Levy(dim)
                            X2 = np.clip(X2, lb, ub)
                            if objf(X2) > fitness:
                                X[i, :] = X2
                    else:  # Hard besiege with progressive rapid dives
                        X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(axis=0))
                        X1 = np.clip(X1, lb, ub)
                        if objf(X1) > fitness:
                            X[i, :] = X1
                        else:
                            X2 = X1 + np.random.randn(dim) * Levy(dim)
                            X2 = np.clip(X2, lb, ub)
                            if objf(X2) > fitness:
                                X[i, :] = X2

        if t % 10 == 0:
            print(f'At iteration {t}, the best fitness is {Rabbit_Energy}')

    return Rabbit_Location, Rabbit_Energy

# Definir límites y dimensiones específicas del problema
lb = np.zeros(5)  # Límites inferiores
ub = np.array([15, 10, 25, 4, 30])  # Límites superiores
dim = 5
SearchAgents_no = 1000
Max_iter = 2000

best_position, best_quality = HHO(objective_function, lb, ub, dim, SearchAgents_no, Max_iter)
print("Best Position:", np.round(best_position).astype(int))
print("Best Quality:", best_quality)
