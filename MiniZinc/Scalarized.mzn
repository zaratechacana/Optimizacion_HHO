% Parámetros
int: c_hat = 6000;
float: w_p = 0;
float: w_q = 1;

% Variables de decisión
var 0..15: x_1;
var 0..10: x_2;
var 0..25: x_3;
var 0..4: x_4;
var 0..30: x_5;

% Cálculo de las funciones objetivo normalizadas
var float: Z_1_norm = (65*x_1 + 90*x_2 + 40*x_3 + 60*x_4 + 20*x_5) / 3175.0;
var float: Z_2_norm = (c_hat - (160*x_1 + 300*x_2 + 40*x_3 + 100*x_4 + 10*x_5)) / c_hat;

% Función objetivo scalarizada
var float: Z = w_q * Z_1_norm + w_p * Z_2_norm;

% Restricciones
constraint 160*x_1 + 300*x_2 <= 3800;
constraint 40*x_3 + 100*x_4 <= 2800;
constraint 40*x_3 + 10*x_5 <= 3500;

% Resolver maximizando Z
solve maximize Z;

output [ "Scalarized Z = " ++ show(Z), 
         "x_1 = " ++ show(x_1),
         "x_2 = " ++ show(x_2),
         "x_3 = " ++ show(x_3),
         "x_4 = " ++ show(x_4),
         "x_5 = " ++ show(x_5),
         "Z_1_norm = " ++ show(Z_1_norm),
         "Z_2_norm = " ++ show(Z_2_norm)];
