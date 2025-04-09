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
    # Penalización si el costo supera 3800
    if cost > 3800:
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
    print(f"Initial Rabbit Location: {Rabbit_Location}, Initial Rabbit Energy: {Rabbit_Energy}")

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

        if t % 1 == 0:
            print(f'At iteration {t}, the best fitness is {Rabbit_Energy}')

    return Rabbit_Location, Rabbit_Energy

# Definir límites y dimensiones específicas del problema
lb = np.zeros(5)  # Límites inferiores
ub = np.array([15, 10, 25, 4, 30])  # Límites superiores
dim = 5
SearchAgents_no = 1000
Max_iter = 1000

best_position, best_quality = HHO(objective_function, lb, ub, dim, SearchAgents_no, Max_iter)
print("Best Position:", best_position)
print("Best Quality:", best_quality)
