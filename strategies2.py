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

        trajet_sonde.append((sonde_x, sonde_y))
        total_iter += 1

        # odeur détectée uniquement si on est dans la fenêtre
        if 0 <= sonde_x < domain_x and 0 <= sonde_y < domain_y:
            if concentration[sonde_y, sonde_x] == 1:
                for _ in range(d):
                    sonde_x -= 1
                    trajet_sonde.append((sonde_x, sonde_y))
                    total_iter += 1
                    if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
                        found = True
                        break
                if found:
                    break

        # succès si on rentre dans la source (même en venant de dehors)
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

        trajet.append((x, y))
        total_iter += 1

        # test source (indépendant des bornes, la source est dans la fenêtre)
        if dans_source(x, y, source_x, source_y, a, b):
            found = True
            break

        # odeur présente seulement si (x,y) est dans la fenêtre
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]
        else:
            c_here = 0

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
            x += np.random.choice([-1, 0, 0, 1])  # un peu plus de chances d'aller à gauche
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

        trajet.append((x, y))
        total_iter += 1

        # test de succès (la source est dans la fenêtre)
        if dans_source(x, y, source_x, source_y, a, b):
            found = True
            break

        # odeur présente ? uniquement si on est dans la fenêtre
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]
        else:
            c_here = 0

        if c_here == 1:
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
