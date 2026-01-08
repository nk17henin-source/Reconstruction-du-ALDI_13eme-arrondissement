import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from strategies4_sans_retour_arriere import strategy_simple, strategy_spiral, strategy_mosquito

# ---- Paramètres de stratégie ----

#strat = "simple"
d = 4                       # Pas de temps caractéristique de la phase de recherche (stratégie simple)

strat = "spiral"

#strat = "zigzag"
#aug_ampl = 1.5

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
    found, trajet_sonde, total_iter = strategy_simple(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, d)
elif strat == "spiral" :
    found, trajet_sonde, total_iter = strategy_spiral(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter)
elif strat == "zigzag" :
    found, trajet_sonde, total_iter = strategy_mosquito(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, aug_ampl)

if found :
    print("Succès")
    print("Nombre total d'itérations = ", total_iter)
else : 
    print("Echec (maximum d'itérations atteint sans trouver la source)")
    
plt.figure(figsize=(12,int(12*l_ratio)))
plt.scatter(x_pts*dl, y_pts*dl-h, c='b', s=5, label='Points concentration')
trajet_x, trajet_y = zip(*trajet_sonde)
trajet_x = np.array(trajet_x)*dl
trajet_y = np.array(trajet_y)*dl-h
plt.plot(trajet_x, trajet_y, 'r.-', label='Trajet sonde')
plt.scatter([trajet_x[0]], [trajet_y[0]], c='red', s=120, marker='*', label='Départ sonde')
rect = plt.Rectangle((source_x*dl, source_y*dl-h), a*dl, b*dl, edgecolor='r', facecolor='none', lw=2, label='Source')
plt.gca().add_patch(rect)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.title("Tracking du CO2 par un moustique - Stratégie : " + strat)
plt.xlim(0, l)
plt.ylim(-h, h)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()




