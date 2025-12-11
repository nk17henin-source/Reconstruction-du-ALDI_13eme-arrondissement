import numpy as np
import matplotlib.pyplot as plt
from strategies2 import strategy_simple, strategy_spiral, strategy_mosquito


domain_x, domain_y = 70, 50
a, b = 6, 4
source_x, source_y = 2, (domain_y-b)//2

# np.random.seed(42)

# --- Champ infotaxis continu + discrétisation binaire ---

V = 2.0        # vitesse vent en x
D = 1.0        # diffusion latérale
tau = 10       # durée de vie

X, Y = np.meshgrid(np.arange(domain_x), np.arange(domain_y))

c = np.zeros_like(X, dtype=float)

downwind = X > source_x
xdist = X - source_x
ydist = Y - source_y  # centre vertical déjà défini

with np.errstate(divide='ignore', invalid='ignore'):
    spread = 4 * D * xdist / V
    in_field = downwind & (spread > 0)
    c[in_field] = np.exp(-xdist[in_field] / (V * tau)) * np.exp(-ydist[in_field]**2 / (2 * spread[in_field]))

# Discrétisation binaire
np.random.seed()  # trajet aléatoire à chaque run
concentration = (np.random.rand(*c.shape) < (c / c.max())).astype(int)

# Zone source forcée à 1
concentration[source_y:source_y+b, source_x:source_x+a] = 1

# Points pour affichage
y_pts, x_pts = np.where(concentration == 1)



max_tot_iter = 3000
start_x, start_y = domain_x-1, np.random.randint(0, domain_y)

# Choix de la stratégie :
# found, trajet_sonde, total_iter = strategy_simple(concentration, source_x, source_y, a, b, start_x, start_y, d=4, max_tot_iter=max_tot_iter)

# found, trajet_sonde, total_iter = strategy_spiral(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter=max_tot_iter)

found, trajet_sonde, total_iter = strategy_mosquito(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter=max_tot_iter)

print("Succès" if found else "Echec (maximum d'itérations atteint sans trouver la source)")

plt.figure(figsize=(12,5))
plt.scatter(x_pts, y_pts, c='b', s=5, label='Points concentration')
trajet_x, trajet_y = zip(*trajet_sonde)
plt.plot(trajet_x, trajet_y, 'r.-', label='Trajet sonde')
plt.scatter([trajet_x[0]], [trajet_y[0]], c='red', s=120, marker='*', label='Départ sonde')
rect = plt.Rectangle((source_x, source_y), a, b, edgecolor='r', facecolor='none', lw=2, label='Source')
plt.gca().add_patch(rect)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Tracking sur champ de concentration infotaxis continu discrétisé')
plt.xlim(0, domain_x)
plt.ylim(0, domain_y)
plt.tight_layout()
plt.show()




