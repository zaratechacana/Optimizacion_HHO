import pulp

# Crear el problema de minimización en PuLP
problem = pulp.LpProblem("Minimize_Advertising_Cost", pulp.LpMinimize)

# Definir las variables. Los límites superior e inferior son según las restricciones y enteros.
x1 = pulp.LpVariable('x1', lowBound=0, upBound=15, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=10, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=25, cat='Integer')
x4 = pulp.LpVariable('x4', lowBound=0, upBound=4, cat='Integer')
x5 = pulp.LpVariable('x5', lowBound=0, upBound=30, cat='Integer')

# Función objetivo para minimizar
problem += 160*x1 + 300*x2 + 40*x3 + 100*x4 + 10*x5, "Total Cost"

# Restricciones según el problema
problem += 160*x1 + 300*x2 <= 3800, "Max TV Budget"
problem += 40*x3 + 100*x4 <= 2800, "Max Newspaper and Magazine Budget"
problem += 40*x3 + 10*x5 <= 3500, "Max Newspaper and Radio Budget"

# Resolver el problema utilizando el solucionador CBC
solver = pulp.PULP_CBC_CMD(msg=False)
problem.solve(solver)

# Imprimir la solución
print("Status:", pulp.LpStatus[problem.status])
print("Minimized Cost:", pulp.value(problem.objective))
print("Values:")
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)
print("x3 =", x3.varValue)
print("x4 =", x4.varValue)
print("x5 =", x5.varValue)

'''
Resultado
Status: Optimal
Minimized Cost: 0.0
Values:
x1 = 0.0
x2 = 0.0
x3 = 0.0
x4 = 0.0
x5 = 0.0
'''
