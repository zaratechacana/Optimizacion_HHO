"2do intento zarate que probe en colab ta de la doga (dog-a) translate it to = (perro-a) == (perra)"

"[1]"
import random
import numpy as np
import math
import scipy.special
"[2]"
domains = {
    "x1": [i for i in range(0, 16)],
    "x2": [i for i in range(0, 11)],
    "x3": [i for i in range(0, 26)],
    "x4": [i for i in range(0, 5)],
    "x5": [i for i in range(0, 31)],
}
"[3]"
constraints = {
    ("x1", "x2"): lambda x1, x2: 164*x1 <= 3800-310*x2,
    ("x2", "x1"): lambda x2, x1: 3800-310*x2 >= 164*x1,
    ("x3", "x4"): lambda x3, x4: 46*x3 <= 2800-111*x4,
    ("x4", "x3"): lambda x4, x3: 2800-111*x4 >= 46*x3,
    ("x3", "x5"): lambda x3, x5: 46*x3 <= 3500-13*x5,
    ("x5", "x3"): lambda x5, x3: 3500-13*x5 >= 46*x3,
}
"[4]"
def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    for x_value in x_domain:
        satisfies = False
        for y_value in y_domain:
            for constraint in all_constraints:
                constraint_func = constraints[constraint]
                if constraint_func(x_value, y_value):
                    satisfies = True
        if not satisfies:
            x_domain.remove(x_value)
            revised = True
    return revised


def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        revised = revise(x, y)
        if revised:
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x]
            queue = queue + neighbors

arcs = [
    ("x1", "x2"),
    ("x2", "x1"),
    ("x3", "x4"),
    ("x4", "x3"),
    ("x3", "x5"),
    ("x5", "x3"),
]

ac3(arcs)

for key, value in domains.items():
    print(f"{key}: {value}\n")


"[7]"
def maximizar(x):
        return 67*x[0]+91*x[1]+43*x[2]+71*x[3]+23*x[4]

"""
CHARP COSAS DE LA ARDILLA, AJUSTAR A COSASA DEL HALCON
#for i in rage(20):
ro=1.204
hg=8
V=5.25
S=154
Gc=1.9
Cl=random.uniform(0.674,1.5)
Cd=0.60
L=1/2*(ro*Cl*(V**2)*S)
D=1/2*(ro*(V**2)*S*Cd)
#print(L,D)
artc=np.arctan(D/L)
#print(artc)
dg=hg/np.tan(artc)
#print(dg)
sf=18
#print(dg/sf)
dg=dg/sf
Pdp=0.1

"""

"[17]"
beta=1.5
maxiter=100
popsize=10
lower_band=[domains[f"x{i+1}"][0] for i in range(len(domains.keys()))]
#CaAg.1106 no se que es esta wea
upper_band=[domains[f"x{i+1}"][-1] for i in range(len(domains.keys()))]
pesos=[164,310,46,111,13]
cap=3000
#lower_band
print(lower_band)

"[10]"
def fs(lowerband,upperband):
     return lowerband + random.uniform(0,1) * (upperband-lowerband)
FS=[np.zeros_like(lower_band).tolist() for i in range(popsize)]
print(FS)

"[11]"
def rho():
    return ((scipy.special.gamma(beta+1)*np.sin((math.pi*beta)/2))/ \
     (scipy.special.gamma((1+beta)/2)*beta*(2)**((beta-1)/2)))**(1/beta)

"[12]"
def levy():
    return 0.01*((random.uniform(0,1)*rho())/(abs(random.uniform(0,1))**(1/beta)))

"[13]"
def randomloc(index):
        return lower_band[index] + levy()*(upper_band[index]-lower_band[index])
print(randomloc(0))



