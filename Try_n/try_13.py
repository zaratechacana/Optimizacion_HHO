import numpy as np

# Definición de los dominios de las variables
domains = {
    'x1': list(range(16)),  # 0 a 15
    'x2': list(range(11)),  # 0 a 10
    'x3': list(range(26)),  # 0 a 25
    'x4': list(range(5)),   # 0 a 4
    'x5': list(range(31))   # 0 a 30
}

# Función de calidad ajustada con los nuevos valores
def quality_function(x1, x2, x3, x4, x5):
    quality_scores = np.array([75, 92.5, 50, 70, 25])  # Puntuaciones promedio de calidad ajustadas
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(quality_scores, quantities)

# Función de costo ajustada con los nuevos valores
def cost_function(x1, x2, x3, x4, x5):
    costs = np.array([180, 325, 60, 110, 15])  # Costos promedio ajustados
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(costs, quantities)

# Restricciones, incluyendo la calidad mínima
epsilon = 5000  # Calidad mínima requerida
constraints = {
    ('x1', 'x2'): lambda x1, x2: 160 * x1 + 300 * x2 <= 3800,
    ('x3', 'x4'): lambda x3, x4: 40 * x3 + 100 * x4 <= 2800,
    ('x3', 'x5'): lambda x3, x5: 40 * x3 + 10 * x5 <= 3500,
    'quality': lambda x1, x2, x3, x4, x5: quality_function(x1, x2, x3, x4, x5) >= epsilon
}

# Función para revisar y actualizar los dominios de acuerdo a todas las restricciones
def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [constraints[constraint] for constraint in constraints if isinstance(constraint, tuple) and constraint[0] == x and constraint[1] == y]
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
        if revise(x, y):
            neighbors = [(nx, ny) for (nx, ny) in arcs if ny == x]
            queue.extend(neighbors)

arcs = [
    ('x1', 'x2'), ('x2', 'x1'), 
    ('x3', 'x4'), ('x4', 'x3'), 
    ('x3', 'x5'), ('x5', 'x3')
]

ac3(arcs)

print("Dominios después de AC3:")
for key, value in domains.items():
    print(f"{key}: {value}")

# Aquí necesitarías implementar el algoritmo de optimización que desees usar para encontrar la mejor solución.
