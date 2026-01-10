import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from strategies5 import strategy_simple, strategy_spiral, strategy_mosquito

# ==========================================
# 1. PARAMÈTRES DE STRATÉGIE
# ==========================================

# Choisissez la stratégie ici : "simple", "spiral", ou "mosquito"
strat = "mosquito" 

# Paramètres communs
d = 4               # Longueur du "Surge" (remontée face au vent quand odeur détectée)

# Paramètres spécifiques
spiral_factor = 1.5 # Facteur de croissance géométrique pour la Spirale
aug_ampl = 1.5      # Facteur de croissance géométrique pour le Mosquito (Zigzag)


# ==========================================
# 2. PARAMÈTRES PHYSIQUES ET DOMAINE
# ==========================================
p = 0.03                    # Probabilité seuil
D = 1                       # Diffusivité [m2/s]
U = 20                      # Vitesse du vent [m/s]
R = 200                     # Intensité de la source [ppm/m3]

n = 100                     # Segmentation du domaine
l_ratio = 1/2               # Ratio hauteur/longueur
cs = 100                    # Seuil de détection

# Constantes calculées
L = 2*D/U                   
l = R/(4*np.pi*D*cs*p)      
h = l*l_ratio/2             
dl = l/n                    
domain_x = n                
domain_y = int(n*l_ratio)   

a, b = 6, 4                 # Dimensions de la source (en pixels)
source_x, source_y = 2, (domain_y-b)//2

# ==========================================
# 3. GÉNÉRATION DU CHAMP DE CONCENTRATION
# ==========================================
X, Y = np.meshgrid(np.arange(domain_x), np.arange(domain_y))
c = np.zeros_like(X, dtype=float)

downwind = X > source_x
xdist = X - source_x
ydist = Y - source_y 
r = np.sqrt(xdist**2 + ydist**2)

with np.errstate(divide='ignore', invalid='ignore'):
    in_field = downwind
    c[in_field] = (p*n/r[in_field])*np.exp(xdist[in_field]*dl/L - r[in_field]*dl/L)

# Discrétisation binaire
np.random.seed() # Aléatoire
concentration = (np.random.rand(*c.shape) < c).astype(int)

# Forcer la source à 1
concentration[source_y:source_y+b+1, source_x:source_x+a+1] = 1

# Points pour affichage
y_pts, x_pts = np.where(concentration == 1)

# ==========================================
# 4. SIMULATION DE LA STRATÉGIE
# ==========================================

max_tot_iter = 3000
start_x, start_y = domain_x-20, np.random.randint(5, domain_y-5)

print(f"Lancement de la stratégie : {strat.upper()}")
print(f"Paramètres : d={d}, Start=({start_x}, {start_y})")

found = False
trajet_sonde = []
total_iter = 0

if strat == "simple":
    found, trajet_sonde, total_iter = strategy_simple(
        concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, 
        d=d
    )

elif strat == "spiral":
    found, trajet_sonde, total_iter = strategy_spiral(
        concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, 
        d=d, 
        spiral_factor=spiral_factor
    )

elif strat == "mosquito": # Note: j'ai unifié le nom "mosquito" ici
    found, trajet_sonde, total_iter = strategy_mosquito(
        concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, 
        d=d, 
        aug_ampl=aug_ampl
    )

# ==========================================
# 5. RÉSULTATS ET AFFICHAGE
# ==========================================

if found:
    print(f"✅ Succès ! Source trouvée en {total_iter} itérations.")
else: 
    print(f"❌ Échec. Maximum d'itérations ({max_tot_iter}) atteint.")

plt.figure(figsize=(12, int(12*l_ratio)))

# Affichage du nuage d'odeur
plt.scatter(x_pts*dl, y_pts*dl-h, c='b', s=5, alpha=0.3, label='Odeur (Concentration)')

# Affichage du trajet
if len(trajet_sonde) > 0:
    trajet_x, trajet_y = zip(*trajet_sonde)
    trajet_x = np.array(trajet_x)*dl
    trajet_y = np.array(trajet_y)*dl-h
    plt.plot(trajet_x, trajet_y, 'r.-', linewidth=1, markersize=3, label='Trajet sonde')
    plt.scatter([trajet_x[0]], [trajet_y[0]], c='red', s=120, marker='*', zorder=10, label='Départ')

# Affichage de la source
rect = plt.Rectangle((source_x*dl, source_y*dl-h), a*dl, b*dl, 
                     edgecolor='r', facecolor='none', lw=2, label='Source')
plt.gca().add_patch(rect)

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend(loc='upper right')
plt.title(f"Simulation Tracking - Stratégie : {strat.upper()} (d={d})")
plt.xlim(0, l)
plt.ylim(-h, h)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()