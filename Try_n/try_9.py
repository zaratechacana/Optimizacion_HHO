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

# Pesos para cada objetivo
w_cost = 0.7  # Peso para el costo
w_quality = 0.3  # Peso para la calidad

# Función de calidad
def quality_function(x1, x2, x3, x4, x5):
    quality_scores = np.array([75, 92.5, 50, 70, 25])  # Puntuaciones promedio de calidad
    quantities = np.array([x1, x2, x3, x4, x5])
    return np.dot(quality_scores, quantities)

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


# Inicialización para HHO
num_hawks = 1  # Número de halcones
max_iterations = 5  # Número máximo de iteraciones
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

# Evaluar las soluciones iniciales y encontrar la mejor
best_hawk = None
best_quality = -np.inf

# Debug: Mostrar las soluciones iniciales
print("Soluciones iniciales:")
for i in range(num_hawks):
    print(f"Halcón {i}: {hawks[i]}")

# Encontrar la mejor solución inicial
for i in range(num_hawks):
    if constraints(hawks[i]):
        current_quality = quality_function(hawks[i])
        if current_quality > best_quality:
            best_quality = current_quality
            best_hawk = hawks[i]

# Si no se encontró ninguna solución válida, inicializar best_hawk con el primer halcón válido
if best_hawk is None:
    best_hawk = hawks[0]
    best_quality = quality_function(best_hawk)

# Debug: Mostrar la mejor solución inicial encontrada
print(f"Mejor calidad inicial: {best_quality}")
print(f"Mejor halcón inicial: {best_hawk}")

# Función de Lévy
def levy_flight(beta, dim):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

# Bucle principal del algoritmo HHO
for t in range(max_iterations):
    print(f"Inicio de la iteración {t + 1}/{max_iterations}")
    for i in range(num_hawks):
        r1, r2, r3, r4, r5 = np.random.rand(5)
        q = np.random.rand()

        # Fase de exploración
        if q >= 0.5:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
            print(f"Exploración - Halcón {i}: Nueva posición {hawks[i]}")
        else:
            hawks[i] = (best_hawk - hawks.mean(axis=0)) - r3 * (LB + r4 * (UB - LB))
            print(f"Refinamiento - Halcón {i}: Nueva posición {hawks[i]}")

        # Asegurarse de que los valores estén dentro de los límites
        hawks[i] = np.clip(hawks[i], LB, UB)

        # Actualización de la energía
        E0 = 2 * r1 - 1  # Energía inicial
        E = 2 * E0 * (1 - t / max_iterations)  # Energía actual

        # Fase de explotación
        if np.abs(E) >= 1:
            hawks[i] = hawks[np.random.randint(num_hawks)] - r1 * np.abs(hawks[np.random.randint(num_hawks)] - 2 * r2 * hawks[i])
            print(f"Exploitation (high energy) - Halcón {i}: Nueva posición {hawks[i]}")
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
            print(f"Exploitation (low energy) - Halcón {i}: Nueva posición {hawks[i]}")

        # Asegurarse de que los valores estén dentro de los límites después de la fase de explotación
        hawks[i] = np.clip(hawks[i], LB, UB)

    # Evaluar la mejor solución
    for i in range(num_hawks):
        if constraints(hawks[i]):  # Solo considerar soluciones que cumplan con las restricciones
            current_quality = quality_function(hawks[i])
            if current_quality > best_quality:
                best_quality = current_quality
                best_hawk = hawks[i]

        # Debug: Mostrar la calidad y configuración del mejor halcón en cada iteración
        print(f"Fin de la iteración {t + 1}: Mejor calidad de anuncios hasta ahora: {best_quality}, Configuración del mejor halcón: {best_hawk}")

# Resultados
print("Mejor calidad de anuncios:", best_quality)
print("Configuración de anuncios:", best_hawk)
