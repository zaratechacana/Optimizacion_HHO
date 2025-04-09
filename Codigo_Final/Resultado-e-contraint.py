import pulp

# Definición del problema
problem = pulp.LpProblem("Maximizar_Calidad", pulp.LpMaximize)

# Definición de variables
x1 = pulp.LpVariable('x1', lowBound=0, upBound=15, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=10, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=25, cat='Integer')
x4 = pulp.LpVariable('x4', lowBound=0, upBound=4, cat='Integer')
x5 = pulp.LpVariable('x5', lowBound=0, upBound=30, cat='Integer')

# Coeficientes de calidad
q1 = 85
q2 = 95
q3 = 60
q4 = 80
q5 = 30

# Función objetivo
problem += q1 * x1 + q2 * x2 + q3 * x3 + q4 * x4 + q5 * x5, "Calidad_Total"

# Coeficientes de costo
c1 = 200
c2 = 350
c3 = 80
c4 = 120
c5 = 20

# Restricciones de costo total
problem += c1 * x1 + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5 <= 9000, "Costo_Total"

# Restricciones adicionales de costos en TV
problem += c1 * x1 + c2 * x2 <= 3800, "Costo_TV"

# Restricciones de costo en diarios y revistas
problem += c3 * x3 + c4 * x4 <= 2800, "Costo_Diarios_Revistas"

# Restricciones de costo en diarios y radio
problem += c3 * x3 + c5 * x5 <= 3500, "Costo_Diarios_Radio"

# Resolver el problema
problem.solve()

# Valores de q
q1_value = q1
q2_value = q2
q3_value = q3
q4_value = q4
q5_value = q5

# Resultados
print(f"x1 = {pulp.value(x1)}")
print(f"x2 = {pulp.value(x2)}")
print(f"x3 = {pulp.value(x3)}")
print(f"x4 = {pulp.value(x4)}")
print(f"x5 = {pulp.value(x5)}")
print(f"q1 = {q1_value}")
print(f"q2 = {q2_value}")
print(f"q3 = {q3_value}")
print(f"q4 = {q4_value}")
print(f"q5 = {q5_value}")
print(f"Calidad Total = {pulp.value(problem.objective)}")


'''
Resultado
x_1 = 15.0
x_2 = 2.0
x_3 = 25.0
x_4 = 4.0
x_5 = 30.0
q1 = 85
q2 = 95
q3 = 60
q4 = 80
q5 = 30
Calidad Total = 4185.0
'''