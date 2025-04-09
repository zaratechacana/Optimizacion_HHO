import pulp

# Crear el problema de maximización en PuLP
problem = pulp.LpProblem("Maximize_Z_weighted", pulp.LpMaximize)

# Definir las variables. Las variables x1 a x5 están sujetas a límites superiores dados y son continuas.
x1 = pulp.LpVariable('x1', lowBound=0, upBound=15, cat='Continuous')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=10, cat='Continuous')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=25, cat='Continuous')
x4 = pulp.LpVariable('x4', lowBound=0, upBound=4, cat='Continuous')
x5 = pulp.LpVariable('x5', lowBound=0, upBound=30, cat='Continuous')

# Ponderaciones ajustables para la función objetivo
w_quality = 1 #Valor entre 0 y 1
w_cost = 0 #Valor entre 0 y 1

# Función objetivo
problem += (w_quality * (65*x1 + 90*x2 + 40*x3 + 60*x4 + 20*x5) 
            - w_cost * (160*x1 + 300*x2 + 40*x3 + 100*x4 + 10*x5)), "Weighted Objective"

# Restricciones
problem += 160*x1 + 300*x2 <= 3800, "Max TV Budget"
problem += 40*x3 + 100*x4 <= 2800, "Max Newspaper and Magazine Budget"
problem += 40*x3 + 10*x5 <= 3500, "Max Newspaper and Radio Budget"

# Resolver el problema
solver = pulp.PULP_CBC_CMD(msg=True)  # msg=True para ver mensajes del solucionador
status = problem.solve(solver)

# Imprimir el estado de la solución y el valor de la función objetivo
print("Status:", pulp.LpStatus[status])
print("Objective Value (Max Z):", pulp.value(problem.objective))

# Imprimir los valores de las variables
print("Solution:")
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)
print("x3 =", x3.varValue)
print("x4 =", x4.varValue)
print("x5 =", x5.varValue)


'''
Objective Value (Max Z): 3235.000003
Solution:
x1 = 15.0
x2 = 4.6666667
x3 = 25.0
x4 = 4.0
x5 = 30.0
'''