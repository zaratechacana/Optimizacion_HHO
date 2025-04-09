% Parámetros
int: c_hat = 6000;

% Variables de decisión
var 0..15: x_1;
var 0..10: x_2;
var 0..25: x_3;
var 0..4: x_4;
var 0..30: x_5;

% Funciones objetivo
var int: Z_1 = 65*x_1 + 85*x_2 + 40*x_3 + 60*x_4 + 20*x_5;
var int: Z_2 = 160*x_1 + 300*x_2 + 40*x_3 + 100*x_4 + 10*x_5;

% Restricciones
constraint 160*x_1 + 300*x_2 <= 3800;
constraint 40*x_3 + 100*x_4 <= 2800;
constraint 40*x_3 + 10*x_5 <= 3500;

% Resolver ambos objetivos, uno como maximización y el otro como minimización
solve maximize Z_1 - Z_2;

output [ "Max Z_1 = " ++ show(Z_1) ++ "\nMin Z_2 = " ++ show(Z_2)];
