import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns

# =============================================================================
# 1. DÉFINITION DES STRATÉGIES (Basé sur strategies4.py)
# =============================================================================

def dans_source(x, y, source_x, source_y, a, b):
    return (source_x <= x <= source_x + a) and (source_y <= y <= source_y + b)

def strategy_simple(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, d):
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

def strategy_spiral(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, d, spiral_factor):
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    mode = "search"
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
        elif mode == "upwind":
            for _ in range(d):
                x -= 1
                y += np.random.choice([-1, 0, 1])
                trajet.append((x, y))
                total_iter += 1
                if dans_source(x, y, source_x, source_y, a, b): found = True; break
            if found: break
        
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y: c_here = concentration[y, x]
        if c_here == 1:
            if mode == "search": mode = "upwind"
        else:
            if mode == "upwind":
                mode = "search" # Restart search HERE
                dir_index = 0; step_length = 1; steps_done_in_segment = 0; segments_done_with_this_length = 0
    return found, trajet, total_iter

def strategy_mosquito(concentration, source_x, source_y, a, b, start_x, start_y, max_tot_iter, d, aug_ampl):
    domain_y, domain_x = concentration.shape
    x, y = start_x, start_y
    trajet = [(x, y)]
    total_iter = 0
    found = False
    mode = "search"
    casting_ampl = 1; casting_dir = 1; casting_debut = True; y_debut_casting = y

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
               x -= 1; y_debut_casting = y; casting_debut = False
            else:
                y += casting_dir
                if abs(y - y_debut_casting) >= casting_ampl:
                    casting_debut = True; casting_dir *= -1; casting_ampl *= aug_ampl
            trajet.append((x, y))
            total_iter += 1
            if dans_source(x, y, source_x, source_y, a, b): found = True; break
        
        c_here = 0
        if 0 <= x < domain_x and 0 <= y < domain_y: c_here = concentration[y, x]
        if c_here == 1:
            if mode != "upwind": mode = "upwind"
        else:
            if mode == "upwind":
                mode = "casting"; casting_ampl = 4; casting_dir = 1; casting_debut = True; y_debut_casting = y
    return found, trajet, total_iter

# =============================================================================
# 2. GÉNÉRATION ET VISUALISATION DES CHAMPS
# =============================================================================

def generate_env(density_type="normal", seed=42):
    D, U, R = 1, 20, 200
    n = 100; l_ratio = 1/2
    p = 0.03 if density_type == "dense" else 0.015 # Densité variable

    cs = 100; L = 2*D/U; l = R/(4*np.pi*D*cs*p)
    dl = l/n; domain_x = n; domain_y = int(n*l_ratio)
    a, b = 6, 4; source_x, source_y = 2, (domain_y-b)//2
    
    X, Y = np.meshgrid(np.arange(domain_x), np.arange(domain_y))
    c = np.zeros_like(X, dtype=float)
    xdist = X - source_x; ydist = Y - source_y; r = np.sqrt(xdist**2 + ydist**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        downwind = X > source_x
        c[downwind] = (p*n/r[downwind])*np.exp(xdist[downwind]*dl/L - r[downwind]*dl/L)
    
    np.random.seed(seed) 
    concentration = (np.random.rand(*c.shape) < c).astype(int)
    concentration[source_y:source_y+b+1, source_x:source_x+a+1] = 1
    return concentration, source_x, source_y, a, b, domain_x, domain_y, dl

# --- Figure 1 : Les Champs de Concentration ---
print("Génération des cartes de concentration...")
fig_fields, axes_fields = plt.subplots(1, 2, figsize=(12, 5))

# Champ Dense
conc_dense, _, _, _, _, _, _, dl_d = generate_env("dense")
y_pts_d, x_pts_d = np.where(conc_dense == 1)
axes_fields[0].scatter(x_pts_d, y_pts_d, c='blue', s=2, alpha=0.5)
axes_fields[0].set_title("DENSE Environnement (p=0.05)")
axes_fields[0].set_aspect('equal')
axes_fields[0].set_xlim(0, 100); axes_fields[0].set_ylim(0, 50)

# Champ Épars
conc_epars, _, _, _, _, _, _, dl_e = generate_env("epars")
y_pts_e, x_pts_e = np.where(conc_epars == 1)
axes_fields[1].scatter(x_pts_e, y_pts_e, c='green', s=2, alpha=0.5)
axes_fields[1].set_title("SPARSE Environnement (p=0.015)")
axes_fields[1].set_aspect('equal')
axes_fields[1].set_xlim(0, 100); axes_fields[1].set_ylim(0, 50)

plt.tight_layout()
plt.show() # Affiche les champs

# =============================================================================
# 3. BOUCLE DE SIMULATION (Statistiques)
# =============================================================================

N_SIMS = 100       # Nombre de simulations par point (Augmenter à 100+ pour plus de précision)
MAX_ITER = 3000

# Paramètres à varier
list_d = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 30]                   # Axe X des graphes
list_spiral_factor = [1.2, 1.5, 2.0]    # Lignes pour Spiral
list_aug_ampl = [1.2, 1.5, 2.0]         # Lignes pour Mosquito

results = []

print(f"\nDémarrage des simulations (N={N_SIMS} par config)...")
start_time = time.time()

for env_name in ["Dense", "Epars"]:
    # On régénère le meme env que sur la figure (meme seed)
    concentration, sx, sy, wa, wb, dx, dy, _ = generate_env(env_name.lower())
    start_x_fixed = dx - 10
    
    # 1. SIMPLE
    for d in list_d:
        iters = []
        for _ in range(N_SIMS):
            sy_start = np.random.randint(5, dy-5)
            f, _, t = strategy_simple(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, d)
            if f: iters.append(t)
        # Score P10
        score = np.percentile(iters, 10) if iters else MAX_ITER
        results.append({"Env": env_name, "Strat": "Simple", "d": d, "Param2": "N/A", "P10": score})
    
    # 2. SPIRAL (d vs spiral_factor)
    for d in list_d:
        for sf in list_spiral_factor:
            iters = []
            for _ in range(N_SIMS):
                sy_start = np.random.randint(5, dy-5)
                f, _, t = strategy_spiral(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, d, sf)
                if f: iters.append(t)
            score = np.percentile(iters, 10) if iters else MAX_ITER
            results.append({"Env": env_name, "Strat": "Spiral", "d": d, "Param2": str(sf), "P10": score})
            
    # 3. MOSQUITO (d vs aug_ampl)
    for d in list_d:
        for aa in list_aug_ampl:
            iters = []
            for _ in range(N_SIMS):
                sy_start = np.random.randint(5, dy-5)
                f, _, t = strategy_mosquito(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, d, aa)
                if f: iters.append(t)
            score = np.percentile(iters, 10) if iters else MAX_ITER
            results.append({"Env": env_name, "Strat": "Mosquito", "d": d, "Param2": str(aa), "P10": score})

    print(f"Environnement {env_name} terminé.")

print(f"Simulation terminée en {time.time()-start_time:.1f}s")

# =============================================================================
# 4. VISUALISATION DES STATISTIQUES
# =============================================================================
df = pd.DataFrame(results)

# Création de la figure de résultats
# 2 Lignes (Dense/Epars) x 3 Colonnes (Simple/Spiral/Mosquito)
fig_res, axes_res = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

# Définition des titres
cols = ["Simple", "Spiral", "Mosquito"]
rows = ["Dense", "Epars"]

for r, env in enumerate(rows):
    for c, strat in enumerate(cols):
        ax = axes_res[r, c]
        
        # Filtrer les données
        data = df[(df["Env"] == env) & (df["Strat"] == strat)]
        
        if strat == "Simple":
            # Juste une ligne
            ax.plot(data["d"], data["P10"], marker='o', linestyle='-', color='b', label='Simple')
        else:
            # Plusieurs lignes selon le Param2 (Facteur de croissance)
            # On utilise seaborn pour gérer les couleurs facilement ou boucle manuelle
            unique_params = data["Param2"].unique()
            for p_val in sorted(unique_params):
                sub_data = data[data["Param2"] == p_val]
                label_name = "Factor " + p_val
                ax.plot(sub_data["d"], sub_data["P10"], marker='o', label=label_name)
        
        # Esthétique
        ax.set_title(f"{strat} ({env})")
        ax.set_xlabel(" d (Surge)")
        if c == 0: ax.set_ylabel("Time [iterations]")
        ax.grid(True, alpha=0.3)
        if strat != "Simple": ax.legend(title="Growth Param.")

plt.suptitle("Performances comparatives (P10)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()