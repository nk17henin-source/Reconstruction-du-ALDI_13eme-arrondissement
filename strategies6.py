import numpy as np

def dans_source(x, y, source_x, source_y, a, b):
    return (source_x <= x <= source_x + a) and (source_y <= y <= source_y + b)

def strategy_simple(concentration, source_x, source_y, a, b,
                    start_x, start_y, max_tot_iter, d):
    # La stratégie simple reste inchangée
    domain_y, domain_x = concentration.shape
    sonde_x, sonde_y = start_x, start_y
    trajet_sonde = [(sonde_x, sonde_y)]
    total_iter = 0
    found = False

    while (sonde_x > 0 and not found and total_iter < max_tot_iter):
        sonde_x += np.random.choice([-1, 0, 1])
        sonde_y += np.random.choice([-1, 0, 1])
        trajet_sonde.append((sonde_x, sonde_y))
        total_iter += 1

        if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
            found = True; break

        if 0 <= sonde_x < domain_x and 0 <= sonde_y < domain_y:
            if concentration[sonde_y, sonde_x] == 1:
                for _ in range(d):
                    sonde_x -= 1
                    trajet_sonde.append((sonde_x, sonde_y))
                    total_iter += 1
                    if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
                        found = True; break
                if found: break

    return found, trajet_sonde, total_iter


def strategy_spiral(concentration, source_x, source_y, a, b,
                    start_x, start_y, max_tot_iter, d, spiral_factor=1.5, Amel=10):
    """
    Si distance > Amel : On RE-DEMARRE une nouvelle spirale à la position actuelle.
    """
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    
    # Modes : "search" (aléatoire initial), "upwind", "spiral"
    mode = "search"
    
    # Variables pour la spirale
    spiral_center_x, spiral_center_y = x, y
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_index = 0
    step_length = 1
    steps_done_in_segment = 0
    segments_done_with_this_length = 0

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            x += np.random.choice([-1, 0, 1])
            y += np.random.choice([-1, 0, 1])
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        elif mode == "upwind":
            for _ in range(d):
                x -= 1
                y += np.random.choice([-1, 0, 1])
                trajet.append((x, y))
                total_iter += 1
                if dans_source(x, y, source_x, source_y, a, b): found = True; break
            if found: break

        elif mode == "spiral":
            # 1. Avancer selon la logique spirale
            dx, dy = directions[dir_index]
            x += dx
            y += dy
            steps_done_in_segment += 1

            if steps_done_in_segment >= int(step_length):
                steps_done_in_segment = 0
                dir_index = (dir_index + 1) % 4
                segments_done_with_this_length += 1
                if segments_done_with_this_length == 2:
                    segments_done_with_this_length = 0
                    step_length *= spiral_factor
                    if step_length < 1: step_length = 1

            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

            # 2. Vérification AMEL : Re-démarrage de la spirale
            dist_from_center = np.sqrt((x - spiral_center_x)**2 + (y - spiral_center_y)**2)
            
            if dist_from_center > Amel:
                # RE-INITIALISATION sur place (Chain of Spirals)
                spiral_center_x, spiral_center_y = x, y
                step_length = 1
                dir_index = 0
                steps_done_in_segment = 0
                segments_done_with_this_length = 0
                # Le mode reste "spiral", on recommence juste la géométrie

        # --- CHECK ODEUR ---
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            if mode != "upwind":
                mode = "upwind"
        else:
            if mode == "upwind":
                # Perdu après Surge -> Démarrage Spirale
                mode = "spiral"
                spiral_center_x, spiral_center_y = x, y
                dir_index = 0
                step_length = 1
                steps_done_in_segment = 0
                segments_done_with_this_length = 0

    return found, trajet, total_iter


def strategy_mosquito(concentration, source_x, source_y, a, b,
                      start_x, start_y, max_tot_iter, d, aug_ampl, Amel=10):
    """
    Si distance > Amel : On RE-DEMARRE le motif de Casting (ZigZag) à la position actuelle.
    """
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    
    # Modes : "search", "upwind", "casting"
    mode = "search"

    # Variables Casting
    initial_ampl = 4 # Amplitude de base pour le restart
    casting_ampl = initial_ampl
    casting_dir = 1
    casting_debut = True
    y_debut_casting = y
    casting_start_x = x
    casting_start_y = y

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            x += np.random.choice([-1, 0, 0, 1])
            y += np.random.choice([-1, 0, 1])
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        elif mode == "upwind":
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

            # 2. Vérification AMEL : Re-démarrage du casting
            dist_from_start = np.sqrt((x - casting_start_x)**2 + (y - casting_start_y)**2)
            
            if dist_from_start > Amel:
                # RE-INITIALISATION sur place (Chain of Castings)
                casting_start_x, casting_start_y = x, y
                y_debut_casting = y
                casting_ampl = initial_ampl # On remet l'amplitude petite
                casting_dir = 1
                casting_debut = True
                # Le mode reste "casting"

        # --- CHECK ODEUR ---
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            if mode != "upwind":
                mode = "upwind"
        else:
            if mode == "upwind":
                # Perdu après Surge -> Démarrage Casting
                mode = "casting"
                casting_start_x, casting_start_y = x, y
                casting_ampl = initial_ampl
                casting_dir = 1
                casting_debut = True
                y_debut_casting = y

    return found, trajet, total_iter