import numpy as np
import matplotlib.pyplot as plt
from strategies_amel import strategy_simple, strategy_spiral, strategy_mosquito


domain_x, domain_y = 70, 50
a, b = 10, 6
source_x, source_y = 2, (domain_y-b)//2

# np.random.seed(42)

# Paramètre Poisson en fonction de la distance à la source (décrément exponentiel)
base_lambda = 8       # Forte concentration proche de la source
k_decay = 0.03        # Vitesse de chute de lambda vers la droite
s_spread = 0.4        # Taux d'élargissement du drapeau
x_pts, y_pts = [], []

# Place la source (rectangle à gauche)
for x in range(source_x, source_x+a):
    for y in range(source_y, source_y+b):
        x_pts.append(x)
        y_pts.append(y)

# Drapeau par loi de Poisson à chaque x à droite
for x in range(source_x + a, domain_x):
    lambda_x = base_lambda * np.exp(-k_decay * (x - (source_x + a)))
    n_particles = np.random.poisson(lambda_x)
    # Largeur croissante (ici traînée de plus en plus large)
    spread = int(b/2 + (x - (source_x+a))*s_spread)  # 0.5 ajuste le taux d'élargissement
    y_center = source_y + b//2
    y_min = max(0, y_center - spread)
    y_max = min(domain_y, y_center + spread)
    for _ in range(n_particles):
        y = np.random.randint(y_min, y_max)
        x_pts.append(x)
        y_pts.append(y)


concentration = np.zeros((domain_y, domain_x), dtype=int)
for x, y in zip(x_pts, y_pts):
    concentration[y, x] = 1


max_tot_iter = 3000
start_x, start_y = domain_x-1, np.random.randint(0, domain_y)

# Choix de la stratégie :
#found, trajet_sonde, total_iter = strategy_simple(concentration, source_x, source_y, a, b, start_x, start_y, d=4, max_tot_iter=max_tot_iter)

found, trajet_sonde, total_iter = strategy_spiral(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter=max_tot_iter)

#found, trajet_sonde, total_iter = strategy_mosquito(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter=max_tot_iter)

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
plt.title('Tracking sur champ de concentration Poisson "drapeau"')
plt.xlim(0, domain_x)
plt.ylim(0, domain_y)
plt.tight_layout()
plt.show()


import numpy as np


def dans_source(x, y, source_x, source_y, a, b):
    return (source_x <= x < source_x + a) and (source_y <= y < source_y + b)


def strategy_simple(concentration, source_x, source_y, a, b,
                    start_x, start_y, d=4, max_tot_iter=3000):
    domain_y, domain_x = concentration.shape

    sonde_x, sonde_y = start_x, start_y
    trajet_sonde = [(sonde_x, sonde_y)]
    total_iter = 0
    found = False

    while (sonde_x > 0
           and not found
           and not dans_source(sonde_x, sonde_y, source_x, source_y, a, b)
           and total_iter < max_tot_iter):

        sonde_x += np.random.choice([-1, 0, 1])
        sonde_y += np.random.choice([-1, 0, 1])

        sonde_x = min(max(sonde_x, 0), domain_x-1)
        sonde_y = min(max(sonde_y, 0), domain_y-1)

        trajet_sonde.append((sonde_x, sonde_y))
        total_iter += 1

        if concentration[sonde_y, sonde_x] == 1:
            for _ in range(d):
                sonde_x -= 1
                sonde_x = max(sonde_x, 0)
                trajet_sonde.append((sonde_x, sonde_y))
                total_iter += 1
                if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
                    found = True
                    break

        if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
            found = True
            break

    return found, trajet_sonde, total_iter



def strategy_spiral(concentration, source_x, source_y, a, b,
                           start_x, start_y, T_loss=10, max_tot_iter=3000):
    """
    SEARCH : spirale carrée qui grandit autour d'un centre (cx, cy)
    UPWIND : remonte le vent (vers la gauche) tant qu'il sent l'odeur.
             Si plus d'odeur pendant T_loss pas -> retour à SEARCH autour
             du dernier point de détection.
    """
    domain_y, domain_x = concentration.shape

    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False

    mode = "search"

    # spirale : expanding square autour de (cx, cy)
    cx, cy = x, y
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # droite, haut, gauche, bas
    dir_index = 0
    step_length = 1
    steps_done_in_segment = 0
    segments_done_with_this_length = 0

    # upwind
    steps_since_last_detection = 0
    last_detection = (x, y)

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            # un pas de spirale carrée autour de (cx, cy)
            dx, dy = directions[dir_index]
            x += dx
            y += dy
            steps_done_in_segment += 1

            if steps_done_in_segment >= step_length:
                steps_done_in_segment = 0
                dir_index = (dir_index + 1) % 4
                segments_done_with_this_length += 1
                if segments_done_with_this_length == 2:
                    segments_done_with_this_length = 0
                    step_length += 1

        elif mode == "upwind":
            # remonter le vent : aller vers la gauche + petit zigzag vertical
            x -= 1
            y += np.random.choice([-1, 0, 1])

        # bornes domaine
        x = min(max(x, 0), domain_x-1)
        y = min(max(y, 0), domain_y-1)

        trajet.append((x, y))
        total_iter += 1

        # test source
        if dans_source(x, y, source_x, source_y, a, b):
            found = True
            break

        c_here = concentration[y, x]

        if mode == "search":
            if c_here == 1:
                # on vient de rencontrer la plume
                mode = "upwind"
                last_detection = (x, y)
                steps_since_last_detection = 0

        elif mode == "upwind":
            if c_here == 1:
                # toujours dans la plume
                last_detection = (x, y)
                steps_since_last_detection = 0
            else:
                steps_since_last_detection += 1
                if steps_since_last_detection >= T_loss:
                    # plume perdue -> nouvelle spirale autour du dernier point d'odeur
                    mode = "search"
                    cx, cy = last_detection
                    x, y = cx, cy
                    trajet.append((x, y))
                    # reset de la spirale
                    dir_index = 0
                    step_length = 1
                    steps_done_in_segment = 0
                    segments_done_with_this_length = 0

    return found, trajet, total_iter




def strategy_mosquito(concentration, source_x, source_y, a, b,
                      start_x, start_y, max_tot_iter=3000):
    
    domain_y, domain_x = concentration.shape

    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False

    mode = "search"


    # paramètres de casting
    casting_ampl = 1
    casting_dir = 1  # +1 ou -1

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            # marche tortueuse autour de la zone, léger biais vers l'amont (gauche)
            x += np.random.choice([-1, 0, 0, 1])   # un peu plus de chances d'aller à gauche
            y += np.random.choice([-1, 0, 1])

        elif mode == "upwind":
            # remonter le vent : x diminue, petit bruit en y
            x -= 1
            y += np.random.choice([-1, 0, 1])

        elif mode == "casting":
            # zigzag vertical perpendiculaire au vent + léger upwind
            y += casting_dir * casting_ampl
            x -= 1
            casting_dir *= -1  # on inverse le sens du zigzag

        # rester dans les bornes
        x = min(max(x, 0), domain_x - 1)
        y = min(max(y, 0), domain_y - 1)

        trajet.append((x, y))
        total_iter += 1

        # test de succès
        if dans_source(x, y, source_x, source_y, a, b):
            found = True
            break

        # odeur présente ?
        if concentration[y, x] == 1:
            if mode == "search":
                mode = "upwind"
            elif mode == "casting":
                mode = "upwind"
            # en upwind, on reste upwind
        else:
            # pas d’odeur
            if mode == "upwind":
                # on vient de la perdre → casting
                mode = "casting"
                casting_ampl = 1
                casting_dir = 1
            elif mode == "casting":
                # toujours rien → on augmente l’amplitude
                casting_ampl += 1
                # (optionnel : si casting_ampl trop grande, revenir en search)

    return found, trajet, total_iter


