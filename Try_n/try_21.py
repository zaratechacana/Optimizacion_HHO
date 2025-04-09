import numpy as np
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import sys
from io import StringIO


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

def max_Z1(valores):
    x1, x2, x3, x4, x5 = valores
    return 65*x1 + 90*x2 + 40*x3 + 60*x4 + 20*x5
max_Z1([14,5,25,4,30])

def min_Z2(valores):
    x1, x2, x3, x4, x5 = valores
    return 160*x1 + 300*x2 + 40*x3 + 100*x4 + 10*x5
min_Z2([14,5,25,4,30])

def cumple_restricciones(valores):
    x1, x2, x3, x4, x5 = valores

    # Restricciones individuales de las variables
    if not (0 <= x1 <= 15 and 0 <= x2 <= 10 and 0 <= x3 <= 25 and 0 <= x4 <= 4 and 0 <= x5 <= 30):
        return False

    # Restricciones de presupuesto para diferentes medios
    if not (160*x1 + 300*x2 <= 3800):
        return False
    if not (40*x3 + 100*x4 <= 2800):
        return False
    if not (40*x3 + 10*x5 <= 3500):
        return False

    return True

cumple_restricciones([14,5,25,4,30])

# Ejemplo de uso de las funciones
valores_ejemplo = [10, 5, 20, 3, 25]
max_z1 = max_Z1(valores_ejemplo)
min_z2 = min_Z2(valores_ejemplo)
restricciones = cumple_restricciones(valores_ejemplo)

(max_z1, min_z2, restricciones)

def scalarized_Z(valores, w_p, w_q):
    x1, x2, x3, x4, x5 = valores
    c_hat = 6000

    # Calcular la calidad total ponderada
    calidad_total = (65*x1 + 90*x2 + 40*x3 + 60*x4 + 20*x5) / 3175
    calidad_ponderada = w_q * calidad_total

    # Calcular el costo total y su ponderación
    costo_total = 160*x1 + 300*x2 + 40*x3 + 100*x4 + 10*x5
    costo_ponderado = w_p * ((c_hat - costo_total) / (c_hat - 0))

    # Función objetivo Z
    Z = calidad_ponderada + costo_ponderado
    return Z

# Ejemplo de uso de la función
#valores_ejemplo = [10, 5, 20, 3, 25]
#w_p_ejemplo = 1
#w_q_ejemplo = 0
#resultado = scalarized_Z(valores_ejemplo, w_p_ejemplo, w_q_ejemplo)
#print(resultado)

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

def initialize_population(pop_size, dim, lb, ub):
    return np.random.uniform(lb, ub, (pop_size, dim))

# Función para evaluar la aptitud de la posición de un halcón
def fitness(pos):
    # Esta función deberá adaptarse a tu problema específico
    x1, x2, x3, x4, x5 = pos
    return scalarized_Z([x1, x2, x3, x4, x5], w_p, w_q)

'''
1/(1+e^(-8x+4))
'''
def sigmoide_wacha(x):
    a = 1 / (1 + np.exp(-8*x + 4))
    if a > 0.5:
        a = 1
    else:
        a = 0
    return a

def valor_sigmoideado(arr):
    result = []
    for n in arr:
        aux = n - int(n)
        s = sigmoide_wacha(aux)
        n = int(n) + s
        result.append(n)
    return result

# Prueba con un arreglo de entrada
arr = np.array([11.42862355, 6.57140077, 16.08925247, 3.07572247, 24.43859152])
output = valor_sigmoideado(arr)
print(output)



#Sigfmoide para cambiar solo la parte decimal del valor de forma wacha

#n = 6,91827391
#aux = n - int(n)
#s = sigmoide (aux)
#s = 1 ~=
#n = int(n) + s

# Función del Algoritmo Harris Hawk Optimization (HHO)
def HHO(N, T, dim, w_p, w_q):
    # Inicializar la población de halcones
    lb = np.zeros(dim)
    ub = np.array([15, 10, 25, 4, 30])
    halcones = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
    for i in range(len(halcones)):
        while True:
            if cumple_restricciones(halcones[i]):
                break
            else:
                halcones[i] = np.random.uniform(0, 1, dim) * (ub - lb) + lb
    # Inicializar la mejor ubicación encontrada (conejo)
    X_rabbit = [14, 5, 25, 4, 30]
    fitness_rabbit = float('-inf')

    # Iteraciones del algoritmo
    for t in range(T):
        for i in range(N):
            # Asegurar que los halcones estén dentro de los límites
            halcones[i] = np.clip(halcones[i], lb, ub)
            halcones[i] = valor_sigmoideado(halcones[i])

            # Calcular la aptitud del halcón
            if cumple_restricciones(halcones[i]):
                current_fitness = fitness(halcones[i])
                if current_fitness > fitness_rabbit:
                    X_rabbit = halcones[i].copy()
                    fitness_rabbit = current_fitness

        # Mover cada halcón hacia el conejo o explorar
        for i in range(N):
            E_0 = 2 * random.random() - 1
            J = 2 * (1 - random.random())
            E = 2 * E_0 * (1 - t/T)

            if abs(E) >= 1:
                q = random.random()
                if q >= 0.5:
                    halcones[i] = X_rabbit - np.random.uniform() * np.abs(X_rabbit - 2 * np.random.uniform() * halcones[i])
                    halcones[i] = valor_sigmoideado(halcones[i])
                    '''
                    while True:
                        if cumple_restricciones(halcones[i]):
                            break
                        else:
                            halcones[i] = X_rabbit - np.random.uniform() * np.abs(X_rabbit - 2 * np.random.uniform() * halcones[i])
                    '''
                else:
                    halcones[i] = (X_rabbit - halcones[i]) - np.random.uniform() * (lb + np.random.uniform() * (ub - lb))
                    halcones[i] = valor_sigmoideado(halcones[i])
                    '''
                    while True:
                        if cumple_restricciones(halcones[i]):
                            break
                        else:
                            halcones[i] = (X_rabbit - halcones[i]) - np.random.uniform() * (lb + np.random.uniform() * (ub - lb))
                    '''
            else:
                r = random.random()
                if r >= 0.5:
                    if abs(E) >= 0.5:
                        ΔX = X_rabbit - halcones[i]
                        halcones[i] = ΔX - E * np.abs(J * X_rabbit - halcones[i])
                        halcones[i] = valor_sigmoideado(halcones[i])
                    else:
                        ΔX = X_rabbit - halcones[i]
                        halcones[i] = X_rabbit - E * np.abs(ΔX)
                        halcones[i] = valor_sigmoideado(halcones[i])
                else:
                    if abs(E) >= 0.5:
                        Y = X_rabbit - E * np.abs(J * X_rabbit - halcones[i])
                        Z = Y + np.random.uniform() * Levy(dim)
                        if fitness(Y) < fitness(halcones[i]):
                            halcones[i] = Y
                            halcones[i] = valor_sigmoideado(halcones[i])
                        elif fitness(Z) < fitness(halcones[i]):
                            halcones[i] = Z
                            halcones[i] = valor_sigmoideado(halcones[i])
                    else:
                        Y = X_rabbit - E * np.abs(J * X_rabbit - halcones[i])
                        Z = Y + np.random.uniform() * Levy(dim)
                        if fitness(Y) < fitness(halcones[i]):
                            halcones[i] = Y
                            halcones[i] = valor_sigmoideado(halcones[i])
                        elif fitness(Z) < fitness(halcones[i]):
                            halcones[i] = Z
                            halcones[i] = valor_sigmoideado(halcones[i])

            # Asegurar que los halcones estén dentro de los límites después de moverse
            halcones[i] = np.clip(halcones[i], lb, ub)

        if t % 1 == 0:
            print(scalarized_Z(X_rabbit, w_p, w_q))
    return X_rabbit

# Ejemplo de configuración y ejecución del algoritmo
N = 10  # Tamaño de la población
T = 50 # Número máximo de iteraciones
dim = 5  # Dimensiones del problema
w_p = 0.5  # Peso para el costo
w_q = 0.5  # Peso para la calidad

# Ejecución del HHO
best_position = HHO(N, T, dim, w_p, w_q)
print("Mejor posición encontrada:", best_position)


def perform_multiple_HHO(N, T, dim, C, w_p, w_q, filename):
    results = []
    for _ in range(C):
        result = HHO(N, T, dim, w_p, w_q)
        results.append(result)

    # Crear DataFrame y guardar en CSV
    df = pd.DataFrame(results)
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv(f'data/{filename}', index=False)

# Parámetros del problema
N = 10
T = 50
dim = 5
C = 30

# Escenarios con diferentes pesos
scenarios = [
    (1, 0, "HHO;W_P_1;W_Q_0.csv"),
    (0.25, 0.75, "HHO;W_P_0_25;W_Q_0_75.csv"),
    (0.5, 0.5, "HHO;W_P_0_5;W_Q_0_5.csv"),
    (0.75, 0.25, "HHO;W_P_0_75;W_Q_0_25.csv"),
    (0, 1, "HHO;W_P_0;W_Q_1.csv")
]

# Ejecutar simulaciones para todos los escenarios
outputs = []
for w_p, w_q, filename in scenarios:
    perform_multiple_HHO(N, T, dim, C, w_p, w_q, filename)
    outputs.append(f"Data saved to {filename}")

print(outputs)


def calculate_objectives(filename):
    # Leer los datos del CSV
    df = pd.read_csv(filename)
    # Calcular valores para Z1 y Z2
    min_z2_values = df.apply(min_Z2, axis=1)
    max_z1_values = df.apply(max_Z1, axis=1)
    return min_z2_values, max_z1_values

def plot_pareto_front(filenames):
    plt.figure(figsize=(10, 6))

    for filename in filenames:
        min_z2, max_z1 = calculate_objectives(filename)
        plt.scatter(min_z2, max_z1, label=f'{filename[:-4]}')  # Remueve '.csv' del nombre para la etiqueta

    plt.title('Pareto Frontier')
    plt.xlabel('Min Z2')
    plt.ylabel('Max Z1')
    plt.legend()
    plt.grid(True)
    plt.show()

# Nombres de los archivos CSV generados
filenames = [
    "data/HHO;W_P_1;W_Q_0.csv",
    "data/HHO;W_P_0_25;W_Q_0_75.csv",
    "data/HHO;W_P_0_5;W_Q_0_5.csv",
    "data/HHO;W_P_0_75;W_Q_0_25.csv",
    "data/HHO;W_P_0;W_Q_1.csv"
]

plot_pareto_front(filenames)




def capture_and_save_output(N, T, dim, w_p, w_q, filename):
    # Redirigir la salida estándar
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    # Ejecutar HHO
    HHO(N, T, dim, w_p, w_q)

    # Obtener la salida capturada y restaurar la salida estándar
    output = sys.stdout.getvalue()
    sys.stdout = original_stdout

    # Procesar la salida capturada para extraer los valores de Z
    lines = output.split('\n')
    z_values = [float(line) for line in lines if line.strip()]

    # Guardar los valores en un archivo CSV
    df = pd.DataFrame(z_values, columns=['Z_value'])
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv(f'data/{filename}', index=False)

# Parámetros del algoritmo
N = 10
T = 50
dim = 5
# Capturar la salida de HHO y guardarla en un archivo CSV
w_p = 1
w_q = 0

capture_and_save_output(N, T, dim, w_p, w_q, "Resultado_Scalarized_HHO;W_P_1;W_Q_0.csv")

w_p = 0.75
w_q = 0.25

capture_and_save_output(N, T, dim, w_p, w_q, "Resultado_Scalarized_HHO;W_P_0_75;W_Q_0_25.csv")

w_p = 0.5
w_q = 0.5

capture_and_save_output(N, T, dim, w_p, w_q, "Resultado_Scalarized_HHO;W_P_0_5;W_Q_0_5.csv")

w_p = 0.25
w_q = 0.75

capture_and_save_output(N, T, dim, w_p, w_q, "Resultado_Scalarized_HHO;W_P_0_25;W_Q_0_75.csv")

w_p = 0
w_q = 1

capture_and_save_output(N, T, dim, w_p, w_q, "Resultado_Scalarized_HHO;W_P_0;W_Q_1.csv")


def plot_csv_files(filenames):
    plt.figure(figsize=(10, 6))

    for filename in filenames:
        # Cargar datos desde CSV
        data = pd.read_csv(filename)

        # Extraer valores de Z
        z_values = data['Z_value']

        # Graficar
        plt.plot(z_values, label=filename.split(";")[1])  # Etiqueta con los valores de w_p y w_q

    # Configuraciones del gráfico
    plt.title('Resultados de HHO para diferentes pesos')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de Z')
    plt.legend()
    plt.grid(True)
    plt.show()

# Lista de archivos CSV
filenames = [
    "data/Resultado_Scalarized_HHO;W_P_1;W_Q_0.csv",
    "data/Resultado_Scalarized_HHO;W_P_0_75;W_Q_0_25.csv",
    "data/Resultado_Scalarized_HHO;W_P_0_5;W_Q_0_5.csv",
    "data/Resultado_Scalarized_HHO;W_P_0_25;W_Q_0_75.csv",
    "data/Resultado_Scalarized_HHO;W_P_0;W_Q_1.csv"
]

# Llamar a la función para graficar
plot_csv_files(filenames)