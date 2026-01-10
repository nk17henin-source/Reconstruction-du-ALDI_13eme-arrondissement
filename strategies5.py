import numpy as np

def dans_source(x, y, source_x, source_y, a, b):
    return (source_x <= x <= source_x + a) and (source_y <= y <= source_y + b)

def strategy_simple(concentration, source_x, source_y, a, b,
                    start_x, start_y, max_tot_iter, d):
    domain_y, domain_x = concentration.shape
    sonde_x, sonde_y = start_x, start_y
    trajet_sonde = [(sonde_x, sonde_y)]
    total_iter = 0
    found = False

    while (sonde_x > 0 and not found and total_iter < max_tot_iter):

        # Marche aléatoire
        sonde_x += np.random.choice([-1, 0, 1])
        sonde_y += np.random.choice([-1, 0, 1])

        trajet_sonde.append((sonde_x, sonde_y))
        total_iter += 1

        if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
            found = True; break

        # Détection
        if 0 <= sonde_x < domain_x and 0 <= sonde_y < domain_y:
            if concentration[sonde_y, sonde_x] == 1:
                # SURGE : on avance de 'd' pas
                for _ in range(d):
                    sonde_x -= 1
                    trajet_sonde.append((sonde_x, sonde_y))
                    total_iter += 1
                    if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
                        found = True; break
                if found: break

    return found, trajet_sonde, total_iter


def strategy_spiral(concentration, source_x, source_y, a, b,
                    start_x, start_y, max_tot_iter, d, spiral_factor=1.5):
    """
    Modifié pour inclure :
    - d : longueur du surge (remontée)
    - spiral_factor : croissance géométrique
    - Pas de T_loss (redémarrage immédiat sur place)
    """
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    mode = "search"

    # Paramètres spirale
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_index = 0
    step_length = 1
    steps_done_in_segment = 0
    segments_done_with_this_length = 0

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            dx, dy = directions[dir_index]
            x += dx
            y += dy
            steps_done_in_segment += 1

            # Logique géométrique
            if steps_done_in_segment >= int(step_length):
                steps_done_in_segment = 0
                dir_index = (dir_index + 1) % 4
                segments_done_with_this_length += 1
                if segments_done_with_this_length == 2:
                    segments_done_with_this_length = 0
                    step_length *= spiral_factor # <--- Croissance géométrique
                    if step_length < 1: step_length = 1

            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        elif mode == "upwind":
            # Surge de longueur 'd'
            for _ in range(d):
                x -= 1
                y += np.random.choice([-1, 0, 1])
                trajet.append((x, y))
                total_iter += 1
                if dans_source(x, y, source_x, source_y, a, b): found = True; break
            if found: break

        # Check Odeur
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            if mode == "search":
                mode = "upwind"
        else:
            # Si on ne sent rien (ou plus rien après le surge)
            if mode == "upwind":
                # Perdu -> on redémarre la spirale ICI
                mode = "search"
                dir_index = 0
                step_length = 1
                steps_done_in_segment = 0
                segments_done_with_this_length = 0

    return found, trajet, total_iter


def strategy_mosquito(concentration, source_x, source_y, a, b,
                      start_x, start_y, max_tot_iter, d, aug_ampl):
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    mode = "search"

    casting_ampl = 1
    casting_dir = 1
    casting_debut = True
    y_debut_casting = y

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            x += np.random.choice([-1, 0, 0, 1])
            y += np.random.choice([-1, 0, 1])
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        elif mode == "upwind":
            # Surge de longueur 'd'
            for _ in range(d):
                x -= 1
                y += np.random.choice([-1, 0, 1])
                trajet.append((x, y))
                total_iter += 1
                if dans_source(x, y, source_x, source_y, a, b): found = True; break
            if found: break

        elif mode == "casting":
            if casting_debut:
               x -= 1
               y_debut_casting = y
               casting_debut = False
            else:
                y += casting_dir
                if abs(y - y_debut_casting) >= casting_ampl:
                    casting_debut = True
                    casting_dir *= -1
                    casting_ampl *= aug_ampl
            
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        # Check Odeur
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            if mode != "upwind":
                mode = "upwind"
        else:
            if mode == "upwind":
                # Perdu -> Casting immédiat
                mode = "casting"
                casting_ampl = 4 # Valeur initiale arbitraire pour le zigzag
                casting_dir = 1
                casting_debut = True
                y_debut_casting = y

    return found, trajet, total_iter