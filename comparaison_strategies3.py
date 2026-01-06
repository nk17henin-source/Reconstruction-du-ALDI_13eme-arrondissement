import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from strategies3 import strategy_simple, strategy_spiral, strategy_mosquito


# ---- Paramètres de simulation ----
N = 200 # Nombre de simulations

strat = "simple"
list_para = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# strat = "spiral"

#strat = "zigzag"
#list_para = [1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2]

   
def simulations(N, strat,para) :

    # --- 1. DÉFINITION DES PARAMÈTRES ---

    if strat == "simple" :
        d = para
    elif strat == "zigzag" :
        aug_ampl = para
        
    # ---- Paramètres ----
    p = 0.03                    # probalilité seuil (concentration négligeable au delà)
    D = 1                       # Diffusivité [m2/s]
    U = 20                       # Vitesse du vent [m/s]
    R = 200                     # Intensité de la source [ppm/m3]

    n = 100                     # Segmentation du domaine d'étude
    l_ratio = 1/2               # Ratio hauteur/longueur

    # ---- Constantes et paramètres caractéristiques ----
    cs = 100                    # Concentration seuil de détection par le moustique [ppm]

    L = 2*D/U                   # Longueur caractéristique de dispersion des particules [m]
    l = R/(4*np.pi*D*cs*p)      # Longueur du domaine d'étude [m]   (0 <= x <= l)
    h = l*l_ratio/2             # Hauteur du domaine d'étude [m]    (-h <= y <= h)
    dl = l/n                    # Longueur d'un segment [m]
    domain_x = n                # Longeur adimmensionnée du domaine (0 <= x <= domain_x)
    domain_y = int(n*l_ratio)   # Hauteur adimmensionnée du domaine (0 <= y <= domain_y)

    a, b = 6, 4                 # dimensions de la boîte "succès"
    source_x, source_y = 2, (domain_y-b)//2

    # --- 2. ANALYSE NUMÉRIQUE ---

    success_count = 0
    total_iterations_success = []

    for i in range(1,N) :
        # ---- Création de la carte de concentration ----
        X, Y = np.meshgrid(np.arange(domain_x), np.arange(domain_y))

        c = np.zeros_like(X, dtype=float)

        downwind = X > source_x
        xdist = X - source_x
        ydist = Y - source_y  # centre vertical déjà défini
        r = np.sqrt(xdist**2 + ydist**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            in_field = downwind
            c[in_field] = (p*n/r[in_field])*np.exp(xdist[in_field]*dl/L - r[in_field]*dl/L)

        # Discrétisation binaire
        np.random.seed()  # trajet aléatoire à chaque run
        concentration = (np.random.rand(*c.shape) < c).astype(int)

        # Zone source forcée à 1
        concentration[source_y:source_y+b+1, source_x:source_x+a+1] = 1
    
        # Points pour affichage
        y_pts, x_pts = np.where(concentration == 1)
    

        max_tot_iter = 3000
        start_x, start_y = domain_x-20, np.random.randint(5,domain_y-5)

        # Choix de la stratégie :
        if strat == "simple" :
            found, _, total_iter = strategy_simple(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, d)
        elif strat == "spiral" :
            found, _, total_iter = strategy_spiral(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter)
        elif strat == "zigzag" :
            found, _, total_iter = strategy_mosquito(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, aug_ampl)

        if found :
            success_count += 1
            total_iterations_success.append(total_iter)

    total_iterations_success.sort()

    # Calcul des métriques
    success_rate = (success_count / N) * 100
    if success_rate >= int(N/10) :
        percentile_10 = total_iterations_success[int(N/10)-1]
    else :
        percentile_10 = -1

    return success_rate, percentile_10


success_list = []
percentile_10_list = []
for k in range(len(list_para)):
    para = list_para[k]
    success_rate, percentile_10 = simulations(N, strat, para)
    success_list.append(success_rate)
    percentile_10_list.append(percentile_10)
    print("Simulation ",k," sur ", len(list_para), " : Done")
