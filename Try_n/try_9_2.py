import numpy as np
from scipy.special import gamma


# Inicialización para HHO
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