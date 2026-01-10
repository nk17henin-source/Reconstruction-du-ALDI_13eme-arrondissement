import numpy as np

def dans_source(x, y, source_x, source_y, a, b):
    return (source_x <= x <= source_x + a) and (source_y <= y <= source_y + b)

# La stratégie simple reste inchangée, elle est purement aléatoire + surge
def strategy_simple(concentration, source_x, source_y, a, b,
                    start_x, start_y, max_tot_iter, d):
    domain_y, domain_x = concentration.shape
    sonde_x, sonde_y = start_x, start_y
    trajet_sonde = [(sonde_x, sonde_y)]
    total_iter = 0
    found = False

    while (sonde_x > 0 and not found and total_iter < max_tot_iter):
        # Marche aléatoire (Global Search)
        sonde_x += np.random.choice([-1, 0, 1])
        sonde_y += np.random.choice([-1, 0, 1])

        trajet_sonde.append((sonde_x, sonde_y))
        total_iter += 1

        if dans_source(sonde_x, sonde_y, source_x, source_y, a, b):
            found = True; break

        # Détection
        if 0 <= sonde_x < domain_x and 0 <= sonde_y < domain_y:
            if concentration[sonde_y, sonde_x] == 1:
                # SURGE : on avance de 'd' pas face au vent
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
    Intègre le paramètre Amel: Rayon maximal de la spirale avant abandon.
    Si la spirale s'éloigne de plus de Amel du point de perte d'odeur, on repasse en aléatoire.
    """
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    
    # Modes: "search" (aléatoire), "upwind" (surge), "spiral" (recherche locale)
    mode = "search" 
    
    # Variables pour la spirale
    spiral_center_x, spiral_center_y = x, y
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_index = 0
    step_length = 1
    steps_done_in_segment = 0
    segments_done_with_this_length = 0

    while (not found and total_iter < max_tot_iter):

        # --- COMPORTEMENT SELON LE MODE ---
        
        if mode == "search":
            # Marche aléatoire classique
            x += np.random.choice([-1, 0, 1])
            y += np.random.choice([-1, 0, 1])
            
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break

        elif mode == "upwind":
            # Surge de longueur 'd' (remontée face au vent)
            for _ in range(d):
                x -= 1
                y += np.random.choice([-1, 0, 1]) # Légère gigue latérale
                trajet.append((x, y))
                total_iter += 1
                if dans_source(x, y, source_x, source_y, a, b): found = True; break
            if found: break
            
            # Après le surge, on vérifie l'odeur plus bas.
            # Si on a toujours l'odeur, on restera en upwind (géré par le check odeur).
            # Sinon, on passera en spirale (géré par le check odeur).

        elif mode == "spiral":
            # Logique géométrique de la spirale
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

            # --- VERIFICATION AMEL (Abandon) ---
            dist_from_center = np.sqrt((x - spiral_center_x)**2 + (y - spiral_center_y)**2)
            if dist_from_center > Amel:
                # On a cherché trop loin, on abandonne la spirale -> Retour aléatoire
                mode = "search"

        # --- CHECK ODEUR & TRANSITIONS ---
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            # Si on sent l'odeur, quel que soit le mode, on passe en Surge (Upwind)
            if mode != "upwind":
                mode = "upwind"
        else:
            # Si on ne sent RIEN
            if mode == "upwind":
                # On vient de perdre la trace après un surge -> Démarrage Spirale ICI
                mode = "spiral"
                spiral_center_x, spiral_center_y = x, y # Le centre est le point de perte
                dir_index = 0
                step_length = 1
                steps_done_in_segment = 0
                segments_done_with_this_length = 0
            
            # Si on est déjà en "spiral" ou "search" et qu'on ne sent rien, 
            # on continue simplement la logique en cours (gérée par la boucle while)

    return found, trajet, total_iter


def strategy_mosquito(concentration, source_x, source_y, a, b,
                      start_x, start_y, max_tot_iter, d, aug_ampl, Amel=10):
    """
    Intègre le paramètre Amel: Distance maximale de casting avant abandon.
    """
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    
    # Modes: "search", "upwind", "casting"
    mode = "search"

    # Variables Casting
    casting_ampl = 1
    casting_dir = 1
    casting_debut = True
    y_debut_casting = y
    casting_start_x = x # Pour calculer la distance Amel
    casting_start_y = y

    while (not found and total_iter < max_tot_iter):

        if mode == "search":
            # Random walk biaisé ou pur
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
               x -= 1 # Petit pas en avant pour initier le zigzag
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

            # --- VERIFICATION AMEL (Abandon) ---
            # Distance euclidienne depuis le début du casting
            dist_from_start = np.sqrt((x - casting_start_x)**2 + (y - casting_start_y)**2)
            if dist_from_start > Amel:
                mode = "search" # Abandon -> Retour aléatoire

        # --- CHECK ODEUR & TRANSITIONS ---
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y:
            c_here = concentration[y, x]

        if c_here == 1:
            if mode != "upwind":
                mode = "upwind"
        else:
            if mode == "upwind":
                # Perdu -> Démarrage Casting immédiat ICI
                mode = "casting"
                casting_start_x, casting_start_y = x, y # Point de référence pour Amel
                casting_ampl = 4 
                casting_dir = 1
                casting_debut = True
                y_debut_casting = y

    return found, trajet, total_iter