# Definiciones iniciales
dimension = 5
costs = [160, 300, 40, 100, 10]
values = [65, 90, 40, 60, 20]
domains = [
    (0, 15), # x1
    (0, 10), # x2
    (0, 25), # x3
    (0, 4),  # x4
    (0, 30)  # x5
]

# Restricciones definidas como funciones
def c1(x1, x2):
    return costs[0]*x1 + costs[1]*x2 <= 3800

def c2(x3, x4):
    return costs[2]*x3 + costs[3]*x4 <= 2800

def c3(x3, x5):
    return costs[2]*x3 + costs[4]*x5 <= 3500

def verificar_restricciones(solution):
    return c1(solution[0], solution[1]) and c2(solution[2], solution[3]) and c3(solution[2], solution[4])

# Funciones de ajuste
def fit_maxi(solution):
    return sum(solution[i] * values[i] for i in range(dimension))

def fit_mini(solution):
    return sum(solution[i] * costs[i] for i in range(dimension))

def fit(solution, weights, max_fit, min_fit, c_hat=10000):
    if not verificar_restricciones(solution):
        return float('inf')  # Consideramos una solución no válida como infinitamente mala

    fit_max = fit_maxi(solution)
    fit_min = fit_mini(solution)

    scalarized_fit = 0
    for p in range(len(weights)):
        for q in range(len(weights)):
            if p != q:
                term_max = (fit_max / max_fit) * weights[p]
                term_min = (c_hat - fit_min) / (c_hat - min_fit) * weights[q]
                scalarized_fit += term_max + term_min

    return scalarized_fit

# Ejemplo de uso
solution = [5, 10, 15, 4, 25]  # Ejemplo de solución
weights = [0.3, 0.7, 0.1, 0.1, 0.4]  # Ponderaciones para cada término de la función objetiva
max_fit = 1000  # Valor estimado o conocido como máximo para fit_maxi
min_fit = 500  # Valor estimado o conocido como mínimo para fit_min

resultado = fit(solution, weights, max_fit, min_fit)
print("Valor de la función objetiva ajustada:", resultado)
