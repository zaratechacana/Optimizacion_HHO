import numpy as np
import pandas as pd
import random
import math
import sys
import matplotlib.pyplot as plt
from io import StringIO
import os
import csv
import itertools

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




# Definir las funciones sigmoides y find_y_interval para cada caso

# Dominio x_1 = [0, 15]
def sigmoid_x_1(x):
    return 1 / (1 + np.exp(-0.6 * (x - 7.5)))

def find_y_interval_x_1(x):
    y = sigmoid_x_1(x)
    interval_width = 1 / 16
    for i in range(16):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 15 if y == 1 else None

# Dominio x_2 = [0, 10]
def sigmoid_x_2(x):
    return 1 / (1 + np.exp(-1 * (x - 5)))

def find_y_interval_x_2(x):
    y = sigmoid_x_2(x)
    interval_width = 1 / 11
    for i in range(11):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 10 if y == 1 else None

# Dominio x_3 = [0, 25]
def sigmoid_x_3(x):
    return 1 / (1 + np.exp(-0.35 * (x - 12.5)))

def find_y_interval_x_3(x):
    y = sigmoid_x_3(x)
    interval_width = 1 / 26
    for i in range(26):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 25 if y == 1 else None

# Dominio x_4 = [0, 4]
def sigmoid_x_4(x):
    return 1 / (1 + np.exp(-2.5 * (x - 2)))

def find_y_interval_x_4(x):
    y = sigmoid_x_4(x)
    if 0 <= y < 0.2:
        return 0
    elif 0.2 <= y < 0.4:
        return 1
    elif 0.4 <= y < 0.6:
        return 2
    elif 0.6 <= y < 0.8:
        return 3
    elif 0.8 <= y <= 1:
        return 4
    else:
        return None

# Dominio x_5 = [0, 30]
def sigmoid_x_5(x):
    return 1 / (1 + np.exp(-0.3 * (x - 15)))

def find_y_interval_x_5(x):
    y = sigmoid_x_5(x)
    interval_width = 1 / 31
    for i in range(31):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 30 if y == 1 else None

# Función maestra
def master_sigmoid(arreglo):
    x_1, x_2, x_3, x_4, x_5 = arreglo
    x_1 = find_y_interval_x_1(x_1)
    x_2 = find_y_interval_x_2(x_2)
    x_3 = find_y_interval_x_3(x_3)
    x_4 = find_y_interval_x_4(x_4)
    x_5 = find_y_interval_x_5(x_5)
    return [x_1, x_2, x_3, x_4, x_5]

# Probar la función maestra con un ejemplo
arreglo = [14.321, 0.312, 24.312, 3.321, 29.321]
resultado = master_sigmoid(arreglo)
print(f"El arreglo transformado es: {resultado}")

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
            #halcones[i] = master_sigmoid(halcones[i])
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
                    ##halcones[i] = master_sigmoid(halcones[i])
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
                    #halcones[i] = master_sigmoid(halcones[i])
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
                        #halcones[i] = master_sigmoid(halcones[i])
                        halcones[i] = valor_sigmoideado(halcones[i])
                    else:
                        ΔX = X_rabbit - halcones[i]
                        halcones[i] = X_rabbit - E * np.abs(ΔX)
                        #halcones[i] = master_sigmoid(halcones[i])
                        halcones[i] = valor_sigmoideado(halcones[i])
                else:
                    if abs(E) >= 0.5:
                        Y = X_rabbit - E * np.abs(J * X_rabbit - halcones[i])
                        Z = Y + np.random.uniform() * Levy(dim)
                        if fitness(Y) < fitness(halcones[i]):
                            halcones[i] = Y
                            #halcones[i] = master_sigmoid(halcones[i])
                            halcones[i] = valor_sigmoideado(halcones[i])
                        elif fitness(Z) < fitness(halcones[i]):
                            halcones[i] = Z
                            #halcones[i] = master_sigmoid(halcones[i])
                            halcones[i] = valor_sigmoideado(halcones[i])
                    else:
                        Y = X_rabbit - E * np.abs(J * X_rabbit - halcones[i])
                        Z = Y + np.random.uniform() * Levy(dim)
                        if fitness(Y) < fitness(halcones[i]):
                            halcones[i] = Y
                            #halcones[i] = master_sigmoid(halcones[i])
                            halcones[i] = valor_sigmoideado(halcones[i])
                        elif fitness(Z) < fitness(halcones[i]):
                            halcones[i] = Z
                            ##halcones[i] = master_sigmoid(halcones[i])
                            halcones[i] = valor_sigmoideado(halcones[i])

            # Asegurar que los halcones estén dentro de los límites después de moverse
            halcones[i] = np.clip(halcones[i], lb, ub)
        if t % 1 == 0:
            print(scalarized_Z(X_rabbit, w_p, w_q))
    return X_rabbit

# Ejemplo de configuración y ejecución del algoritmo
N = 5  # Tamaño de la población
T = 30 # Número máximo de iteraciones
dim = 5  # Dimensiones del problema
w_p = 0.3  # Peso para el costo
w_q = 0.7  # Peso para la calidad

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
N = 5
T = 30
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
N = 5
T = 30
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

'''
dominio x_1 = [0,15]
1/(1+e^(-0.6(x-7.5)))
'''

# Definir la función sigmoide
def sigmoid_x_1(x):
    return 1 / (1 + np.exp(-0.6 * (x - 7.5)))

# Dividir el eje y en 16 intervalos iguales y determinar el intervalo de un valor dado
def find_y_interval_x_1(x):
    y = sigmoid_x_1(x)
    interval_width = 1 / 16
    for i in range(16):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 15 if y == 1 else None

# Probar la función con un valor dado
x_value = 3  # Cambia este valor para probar diferentes entradas
interval = find_y_interval_x_1(x_value)
print(f"El valor x = {x_value} cae en el intervalo y = {interval}")

# Graficar la sigmoide y los intervalos en el eje y
x = np.linspace(0, 15, 400)
y = sigmoid_x_1(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoide: 1 / (1 + e^{-0.6(x - 7.5)})')
for i in range(1, 16):
    plt.axhline(y=i * (1 / 16), color='red', linestyle='--')
plt.scatter([x_value], [sigmoid_x_1(x_value)], color='black', zorder=5)
plt.text(x_value, sigmoid_x_1(x_value), f'  ({x_value}, {sigmoid_x_1(x_value):.2f})', color='black')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoide con Intervalos en el Eje y')
plt.legend()
plt.grid(True)
plt.show()

'''
dominio x_2 = [0,10]
1/(1+e^(-1(x-5)))
'''

# Definir la función sigmoide
def sigmoid_x_2(x):
    return 1 / (1 + np.exp(-1 * (x - 5)))

# Dividir el eje y en 11 intervalos iguales y determinar el intervalo de un valor dado
def find_y_interval_x_2(x):
    y = sigmoid_x_2(x)
    interval_width = 1 / 11
    for i in range(11):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 10 if y == 1 else None

# Probar la función con un valor dado
x_value = 6  # Cambia este valor para probar diferentes entradas
interval = find_y_interval_x_2(x_value)
print(f"El valor x = {x_value} cae en el intervalo y = {interval}")

# Graficar la sigmoide y los intervalos en el eje y
x = np.linspace(0, 10, 400)
y = sigmoid_x_2(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoide: 1 / (1 + e^{-1(x - 5)})')
for i in range(1, 11):
    plt.axhline(y=i * (1 / 11), color='red', linestyle='--')
plt.scatter([x_value], [sigmoid_x_2(x_value)], color='black', zorder=5)
plt.text(x_value, sigmoid_x_2(x_value), f'  ({x_value}, {sigmoid_x_2(x_value):.2f})', color='black')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoide con Intervalos en el Eje y')
plt.legend()
plt.grid(True)
plt.show()

'''
dominio x_3 = [0,25]
1/(1+e^(-0.35(x-12.5)))
'''
# Definir la función sigmoide
def sigmoid_x_3(x):
    return 1 / (1 + np.exp(-0.35 * (x - 12.5)))

# Dividir el eje y en 26 intervalos iguales y determinar el intervalo de un valor dado
def find_y_interval_x_3(x):
    y = sigmoid_x_3(x)
    interval_width = 1 / 26
    for i in range(26):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 25 if y == 1 else None

# Probar la función con un valor dado
x_value = 20  # Cambia este valor para probar diferentes entradas
interval = find_y_interval_x_3(x_value)
print(f"El valor x = {x_value} cae en el intervalo y = {interval}")

# Graficar la sigmoide y los intervalos en el eje y
x = np.linspace(0, 25, 400)
y = sigmoid_x_3(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoide: 1 / (1 + e^{-0.35(x - 12.5)})')
for i in range(1, 26):
    plt.axhline(y=i * (1 / 26), color='red', linestyle='--')
plt.scatter([x_value], [sigmoid_x_3(x_value)], color='black', zorder=5)
plt.text(x_value, sigmoid_x_3(x_value), f'  ({x_value}, {sigmoid_x_3(x_value):.2f})', color='black')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoide con Intervalos en el Eje y')
plt.legend()
plt.grid(True)
plt.show()

'''
dominio x_4 = [0,4]
1/(1+e^(-2.5(x-2)))
'''


# Definir la función sigmoide
def sigmoid_x_4(x):
    return 1 / (1 + np.exp(-2.5 * (x - 2)))

# Dividir el eje y en cinco intervalos iguales y determinar el intervalo de un valor dado
def find_y_interval_x_4(x):
    y = sigmoid_x_4(x)
    if 0 <= y < 0.2:
        return 0
    elif 0.2 <= y < 0.4:
        return 1
    elif 0.4 <= y < 0.6:
        return 2
    elif 0.6 <= y < 0.8:
        return 3
    elif 0.8 <= y <= 1:
        return 4
    else:
        return None

# Probar la función con un valor dado
x_value = 3.321  # Cambia este valor para probar diferentes entradas
interval = find_y_interval_x_4(x_value)
print(f"El valor x = {x_value} cae en el intervalo y = {interval}")

# Graficar la sigmoide y los intervalos en el eje y

x = np.linspace(0, 4, 400)
y = sigmoid_x_4(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoide: 1 / (1 + e^{-2.5(x - 2)})')
plt.axhline(y=0.2, color='red', linestyle='--', label='Intervalo 0.2')
plt.axhline(y=0.4, color='orange', linestyle='--', label='Intervalo 0.4')
plt.axhline(y=0.6, color='green', linestyle='--', label='Intervalo 0.6')
plt.axhline(y=0.8, color='blue', linestyle='--', label='Intervalo 0.8')
plt.axhline(y=1.0, color='purple', linestyle='--', label='Intervalo 1.0')
plt.scatter([x_value], [sigmoid_x_4(x_value)], color='black', zorder=5)
plt.text(x_value, sigmoid_x_4(x_value), f'  ({x_value}, {sigmoid_x_4(x_value):.2f})', color='black')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoide con Intervalos en el Eje y')
plt.legend()
plt.grid(True)
plt.show()

'''
dominio x_5 = [0,30]
1/(1+e^(-0.3(x-15)))
'''
# Definir la función sigmoide
def sigmoid_x_5(x):
    return 1 / (1 + np.exp(-0.3 * (x - 15)))

# Dividir el eje y en 31 intervalos iguales y determinar el intervalo de un valor dado
def find_y_interval_x_5(x):
    y = sigmoid_x_5(x)
    interval_width = 1 / 31
    for i in range(31):
        if i * interval_width <= y < (i + 1) * interval_width:
            return i
    return 30 if y == 1 else None

# Probar la función con un valor dado
x_value = 20  # Cambia este valor para probar diferentes entradas
interval = find_y_interval_x_5(x_value)
print(f"El valor x = {x_value} cae en el intervalo y = {interval}")

# Graficar la sigmoide y los intervalos en el eje y
x = np.linspace(0, 30, 400)
y = sigmoid_x_5(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoide: 1 / (1 + e^{-0.3(x - 15)})')
for i in range(1, 31):
    plt.axhline(y=i * (1 / 31), color='red', linestyle='--')
plt.scatter([x_value], [sigmoid_x_5(x_value)], color='black', zorder=5)
plt.text(x_value, sigmoid_x_5(x_value), f'  ({x_value}, {sigmoid_x_5(x_value):.2f})', color='black')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoide con Intervalos en el Eje y')
plt.legend()
plt.grid(True)
plt.show()



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

# Dominios de cada variable
dominios = [
    range(16),  # x1 de 0 a 15
    range(11),  # x2 de 0 a 10
    range(26),  # x3 de 0 a 25
    range(5),   # x4 de 0 a 4
    range(31)   # x5 de 0 a 30
]

# Crear archivo CSV para guardar los resultados
with open('data/pareto.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Escribir el encabezado del CSV
    writer.writerow(['x1', 'x2', 'x3', 'x4', 'x5'])

    # Generar todas las combinaciones posibles
    for combinacion in itertools.product(*dominios):
        if cumple_restricciones(combinacion):
            # Escribir la combinación en el archivo CSV si cumple las restricciones
            writer.writerow(combinacion)



def plot_pareto_front(filenames, specific_values):
    plt.figure(figsize=(10, 6))

    for filename in filenames:
        min_z2, max_z1 = calculate_objectives(filename)
        plt.scatter(min_z2, max_z1, label=f'{filename[:-4]}')  # Remueve '.csv' del nombre para la etiqueta

    # Evaluar el punto específico y mostrarlo en el gráfico
    specific_min_z2 = min_Z2(specific_values)
    specific_max_z1 = max_Z1(specific_values)
    plt.scatter(specific_min_z2, specific_max_z1, color='green', s=100, label='Punto Específico')

    plt.title('Frontera de Pareto')
    plt.xlabel('Min Z2')
    plt.ylabel('Max Z1')
    plt.legend()
    plt.grid(True)
    plt.show()

# Nombres de los archivos CSV generados
filenames = [
    "data/pareto.csv"
]

# Valores específicos para el punto verde
# Ejemplo de configuración y ejecución del algoritmo
N = 10  # Tamaño de la población
T = 50 # Número máximo de iteraciones
dim = 5  # Dimensiones del problema
w_p = 0  # Peso para el costo
w_q = 1  # Peso para la calidad

specific_values = HHO(N, T, dim, w_p, w_q)

plot_pareto_front(filenames, specific_values)