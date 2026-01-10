import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns

# =============================================================================
# 1. IMPORTATION DES STRATÉGIES
# =============================================================================
# Assurez-vous que strategies6.py est dans le même dossier.
try:
    from strategies6 import strategy_simple, strategy_spiral, strategy_mosquito
except ImportError:
    try:
        from strategies6_rdm import strategy_simple, strategy_spiral, strategy_mosquito
    except ImportError:
        print("ERREUR : Impossible d'importer strategies6.py")
        exit()

# =============================================================================
# 2. GÉNÉRATION DE L'ENVIRONNEMENT
# =============================================================================

def generate_env(density_type="normal", seed=42):

    D, p, R = 1, 0.009, 200
    n = 100; l_ratio = 1/2
    
    # Variation de U (Vent)
    if density_type == "dense":
        U = 0.2  # Vent faible -> Panache large
    else:
        U = 2  # Vent fort -> Panache fin/déchiré

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

# =============================================================================
# 3. VISUALISATION DES ENVIRONNEMENTS
# =============================================================================
print("Génération des cartes de concentration pour vérification...")
fig_fields, axes_fields = plt.subplots(1, 2, figsize=(12, 5))

# Champ Dense
conc_dense, _, _, _, _, _, _, dl_d = generate_env("dense")
y_pts_d, x_pts_d = np.where(conc_dense == 1)
axes_fields[0].scatter(x_pts_d, y_pts_d, c='k', s=2, alpha=0.5)
axes_fields[0].set_title("DENSE (U=0.2 m/s)")
axes_fields[0].set_aspect('equal')
axes_fields[0].set_xlim(0, 100); axes_fields[0].set_ylim(0, 50)

# Champ Épars
conc_epars, _, _, _, _, _, _, dl_e = generate_env("epars")
y_pts_e, x_pts_e = np.where(conc_epars == 1)
axes_fields[1].scatter(x_pts_e, y_pts_e, c='k', s=2, alpha=0.5)
axes_fields[1].set_title("SPARSE (U=2 m/s)")
axes_fields[1].set_aspect('equal')
axes_fields[1].set_xlim(0, 100); axes_fields[1].set_ylim(0, 50)

plt.tight_layout()
plt.show() # Affiche la figure immédiatement

# =============================================================================
# 4. MOTEUR DE SIMULATION
# =============================================================================

N_SIMS = 300       # Nombre de simulations par configuration
MAX_ITER = 3000

# --- PARAMÈTRES À VARIER ---
# 1. Surge (Axe X)
list_d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]

# 2. Amel (Distance de redémarrage)
list_amel = [10, 20, 30]

# 3. Facteurs de croissance (Spiral Factor / Aug Ampl)
list_factors = [1.2, 1.5, 2.0]

results = []

print(f"\nDémarrage des simulations complètes (N={N_SIMS})...")
print("Cela peut prendre quelques minutes.")
start_time = time.time()

for env_name in ["Dense", "Epars"]:
    concentration, sx, sy, wa, wb, dx, dy, _ = generate_env(env_name.lower())
    start_x_fixed = dx - 5
    
    for d in list_d:
        # --- SIMPLE (Référence) ---
        # Simple n'a ni Amel ni Factor, on la calcule une fois par 'd'
        successes = []
        times = []
        for _ in range(N_SIMS):
            sy_start = np.random.randint(5, dy-5)
            f, _, t = strategy_simple(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, d)
            if f:
                successes.append(1)
                times.append(t)
            else:
                successes.append(0)
        
        # Calcul stats
        p10 = np.percentile(times, 10) if times else MAX_ITER
        rate = sum(successes) / N_SIMS
        # On loggue avec des valeurs "Ref" pour faciliter le filtrage plus tard
        for am in list_amel:
            for fac in list_factors:
                results.append({
                    "Env": env_name, "Strat": "Simple", "d": d, 
                    "Amel": am, "Factor": fac, # Dummy values pour affichage unifié
                    "P10": p10, "SuccessRate": rate
                })

        # --- SPIRAL ---
        for amel in list_amel:
            for fac in list_factors:
                successes = []
                times = []
                for _ in range(N_SIMS):
                    sy_start = np.random.randint(5, dy-5)
                    f, _, t = strategy_spiral(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, 
                                              d=d, spiral_factor=fac, Amel=amel)
                    if f:
                        successes.append(1)
                        times.append(t)
                    else:
                        successes.append(0)
                
                p10 = np.percentile(times, 10) if times else MAX_ITER
                rate = sum(successes) / N_SIMS
                results.append({
                    "Env": env_name, "Strat": "Spiral", "d": d, 
                    "Amel": amel, "Factor": fac, 
                    "P10": p10, "SuccessRate": rate
                })

        # --- MOSQUITO ---
        for amel in list_amel:
            for fac in list_factors:
                successes = []
                times = []
                for _ in range(N_SIMS):
                    sy_start = np.random.randint(5, dy-5)
                    f, _, t = strategy_mosquito(concentration, sx, sy, wa, wb, start_x_fixed, sy_start, MAX_ITER, 
                                                d=d, aug_ampl=fac, Amel=amel)
                    if f:
                        successes.append(1)
                        times.append(t)
                    else:
                        successes.append(0)
                
                p10 = np.percentile(times, 10) if times else MAX_ITER
                rate = sum(successes) / N_SIMS
                results.append({
                    "Env": env_name, "Strat": "Mosquito", "d": d, 
                    "Amel": amel, "Factor": fac, 
                    "P10": p10, "SuccessRate": rate
                })

    print(f"Environnement {env_name} terminé.")

print(f"Total Simulation Time: {time.time()-start_time:.1f}s")
df = pd.DataFrame(results)

# =============================================================================
# 5. FONCTION DE PLOTTING GÉNÉRIQUE
# =============================================================================

def plot_analysis(df, fixed_param_name, fixed_param_value, varying_param_name, title_prefix):
    """
    Génère 2 figures (P10 et Taux de Succès) en fixant un paramètre et en variant l'autre.
    """
    # Filtrage des données
    data = df[
        (df["Strat"] == "Simple") | 
        (df[fixed_param_name] == fixed_param_value)
    ]
    
    # ---------------- FIGURE P10 (Temps) ----------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    rows = ["Dense", "Epars"]
    cols = ["Simple", "Spiral", "Mosquito"]
    
    for r, env in enumerate(rows):
        for c, strat in enumerate(cols):
            ax = axes1[r, c]
            sub = data[(data["Env"] == env) & (data["Strat"] == strat)]
            
            if strat == "Simple":
                simple_sub = sub.drop_duplicates(subset=["d"])
                ax.plot(simple_sub["d"], simple_sub["P10"], 'b-o', label="Simple")
            else:
                unique_vals = sorted(sub[varying_param_name].unique())
                for val in unique_vals:
                    s = sub[sub[varying_param_name] == val]
                    ax.plot(s["d"], s["P10"], marker='o', label=f"{varying_param_name}={val}")
            
            if r==0: ax.set_title(f"{strat}")
            if c==0: ax.set_ylabel(f"{env}\nTime (P10)")
            if r==1: ax.set_xlabel("d (Surge)")
            ax.grid(True, alpha=0.3)
            if strat != "Simple": ax.legend(fontsize='small', title=varying_param_name)

    fig1.suptitle(f"{title_prefix} - Performance P10", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # ---------------- FIGURE SUCCES RATE ----------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    
    for r, env in enumerate(rows):
        for c, strat in enumerate(cols):
            ax = axes2[r, c]
            sub = data[(data["Env"] == env) & (data["Strat"] == strat)]
            
            if strat == "Simple":
                simple_sub = sub.drop_duplicates(subset=["d"])
                ax.plot(simple_sub["d"], simple_sub["SuccessRate"], 'b-o', label="Simple")
            else:
                unique_vals = sorted(sub[varying_param_name].unique())
                for val in unique_vals:
                    s = sub[sub[varying_param_name] == val]
                    ax.plot(s["d"], s["SuccessRate"], marker='o', label=f"{varying_param_name}={val}")
            
            if r==0: ax.set_title(f"{strat}")
            if c==0: ax.set_ylabel(f"{env}\nSuccess Rate")
            if r==1: ax.set_xlabel("d (Surge)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            if strat != "Simple": ax.legend(fontsize='small', title=varying_param_name)

    fig2.suptitle(f"{title_prefix} - Success Rate", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# =============================================================================
# 6. GÉNÉRATION DES GRAPHES
# =============================================================================

# ANALYSE 1 : Impact de AMEL (avec Facteur fixé à 1.5)
plot_analysis(df, fixed_param_name="Factor", fixed_param_value=1.5, 
              varying_param_name="Amel", title_prefix="Amel impact")

# ANALYSE 2 : Impact du FACTEUR GÉOMÉTRIQUE (avec Amel fixé à 10)
plot_analysis(df, fixed_param_name="Amel", fixed_param_value=10, 
              varying_param_name="Factor", title_prefix="Growth Factor impact")

plt.show()