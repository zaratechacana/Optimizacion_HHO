import pulp

# Definir el problema de maximización usando PuLP
problem = pulp.LpProblem("Maximize_Media_Exposure", pulp.LpMaximize)

# Definir las variables
x1 = pulp.LpVariable('x1', lowBound=0, upBound=15, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=10, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=25, cat='Integer')
x4 = pulp.LpVariable('x4', lowBound=0, upBound=4, cat='Integer')
x5 = pulp.LpVariable('x5', lowBound=0, upBound=30, cat='Integer')

# Función objetivo
problem += 65*x1 + 85*x2 + 40*x3 + 60*x4 + 20*x5, "Objective"

# Restricciones
problem += 160*x1 + 300*x2 <= 3800, "Television Budget"
problem += 40*x3 + 100*x4 <= 2800, "Newspaper and Magazine Budget"
problem += 40*x3 + 10*x5 <= 3500, "Newspaper and Radio Budget"

# Resolver el problema
solver = pulp.PULP_CBC_CMD(msg=False)
problem.solve(solver)

# Mostrar resultados
result = {
    "x1": x1.varValue,
    "x2": x2.varValue,
    "x3": x3.varValue,
    "x4": x4.varValue,
    "x5": x5.varValue,
    "Objective": pulp.value(problem.objective)
}

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
Minimized Cost: 3200.0
Values:
x1 = 14.0
x2 = 5.0
x3 = 25.0
x4 = 4.0
x5 = 30.0
'''
