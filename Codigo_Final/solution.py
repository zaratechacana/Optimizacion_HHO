class solution:
    def __init__(self):
        self.best = float('-inf')  # Al ser un problema de maximización, inicializamos con -inf
        self.bestIndividual = None  # Almacena la mejor solución encontrada
        self.convergence = []  # Registro del mejor valor de la función objetivo en cada iteración
        self.optimizer = "HHO"  # Nombre del optimizador utilizado
        self.objfname = ""  # Nombre de la función objetivo
        self.startTime = ""  # Momento en que inicia la ejecución del algoritmo
        self.endTime = ""  # Momento en que termina la ejecución
        self.executionTime = 0  # Duración total de la ejecución del algoritmo

        # Variables específicas para análisis detallado
        self.totalQuality = 0  # Almacenará la calidad total de la mejor solución
        self.totalCost = 0  # Almacenará el costo total de la mejor solución
