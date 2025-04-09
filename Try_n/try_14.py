import numpy as np
import random
import math
import time

# Definición de los dominios de las variables
domains = {
    'x1': list(range(16)),  # 0 a 15
    'x2': list(range(11)),  # 0 a 10
    'x3': list(range(26)),  # 0 a 25
    'x4': list(range(5)),   # 0 a 4
    'x5': list(range(31))   # 0 a 30
}

# Función de calidad
def quality_function(x):
    quality_scores = np.array([75, 92.5, 50, 70, 25])  # Puntuaciones promedio de calidad
    return np.dot(quality_scores, x)

# Función de costo
def cost_function(x):
    costs = np.array([180, 325, 60, 110, 15])  # Costos promedio
    return np.dot(costs, x)

# Función objetivo para HHO
def objective_function(x):
    quality = quality_function(x)
    cost = cost_function(x)
    if cost > 3800:
        return float('inf')  # Penalización por superar el costo máximo
    return -quality  # Maximizar la calidad negando el valor

# Inicialización y definición de HHO
def HHO(lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = -float("inf")
    
    # Initial location of Harris hawks
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    
    # Main loop
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # Applying simple bounds/limits
            X[i, :] = np.clip(X[i, :], lb, ub)
            
            # Calculate fitness
            fitness = objective_function(X[i, :])
            
            # Update the location of Rabbit
            if fitness > Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
                
        E1 = 2 * (1 - (t / Max_iter))
        
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            
            if abs(Escaping_Energy) >= 1:
                # Exploration
                X_rand = X[np.random.randint(SearchAgents_no), :]
                X[i, :] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X[i, :])
            else:
                # Exploitation
                if random.random() >= 0.5:
                    # Soft besiege
                    Jump_strength = 2 * (1 - np.random.rand())
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                else:
                    # Hard besiege
                    Jump_strength = 2 * (1 - np.random.rand())
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))
        
        if t % 10 == 0:
            print(f"At iteration {t}, the best fitness is {Rabbit_Energy}")

    return Rabbit_Location, -Rabbit_Energy  # Return the best location and its fitness

# Define limits
lb = np.array([0, 0, 0, 0, 0])
ub = np.array([15, 10, 25, 4, 30])
dim = 5
SearchAgents_no = 1000
Max_iter = 1000

# Run HHO
best_position, best_quality = HHO(lb, ub, dim, SearchAgents_no, Max_iter)
print("Best Position:", best_position)
print("Best Quality:", best_quality)
