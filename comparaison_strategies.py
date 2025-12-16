import numpy as np
import matplotlib.pyplot as plt
from strategies import strategy_simple, strategy_spiral, strategy_mosquito 

# --- 1. ENVIRONNEMENT ---
domain_x, domain_y = 70, 50
a, b = 10, 6
source_x, source_y = 2, (domain_y-b)//2

# Paramètres de la Plume
base_lambda = 8       
k_decay = 0.03        
s_spread = 0.4        
x_pts, y_pts = [], []

# Création du champ de concentration
for x in range(source_x, source_x+a):
    for y in range(source_y, source_y+b):
        x_pts.append(x)
        y_pts.append(y)

for x in range(source_x + a, domain_x):
    lambda_x = base_lambda * np.exp(-k_decay * (x - (source_x + a)))
    n_particles = np.random.poisson(lambda_x)
    spread = int(b/2 + (x - (source_x+a))*s_spread) 
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

# Paramètres de simulation
max_tot_iter = 3000
start_x = domain_x-1
start_y_ref = (domain_y-b)//2 # Référence du centre de départ

# --- 2. ANALYSE NUMÉRIQUE ---

# Nous allons utiliser les paramètres optimaux trouvés ou des valeurs standards:
STRATEGIES_TO_TEST = [
    # Hypothèse: d=4 (un saut agressif)
    ("Simple (d=4)", strategy_simple, {'d': 4}),
    
    # Hypothèse: T_loss=10
    ("Spiral (T_loss=10)", strategy_spiral, {'T_loss': 10}), 
    
    # Stratégie Mosquito (pas de paramètre externe à optimiser ici)
    ("Mosquito", strategy_mosquito, {}), 
]

N_SIMULATIONS = 150 # Nombre d'essais pour la robustesse statistique

print("\n--- Comparaison Numérique des Stratégies de Tracking ---")
print(f"Banc d'essai: Plume Poisson Turbulente | N={N_SIMULATIONS} simulations par stratégie")
print("-" * 75)

comparaison_results = []

for name, strategy_func, params in STRATEGIES_TO_TEST:
    
    success_count = 0
    total_iterations_success = 0
    
    for i in range(N_SIMULATIONS):
        # Point de départ légèrement aléatoire pour éviter un biais de position
        start_y = np.random.randint(max(0, start_y_ref - 5), min(domain_y, start_y_ref + 5))
        
        # Exécution de la stratégie (Simulation du tracking du moustique)
        found, _, total_iter = strategy_func(
            concentration, 
            source_x, source_y, a, b, 
            start_x, start_y, 
            max_tot_iter=max_tot_iter,
            **params 
        )
        
        if found:
            success_count += 1
            total_iterations_success += total_iter

    # Calcul des métriques
    success_rate = (success_count / N_SIMULATIONS) * 100
    avg_iter_success = total_iterations_success / success_count if success_count > 0 else np.nan
    
    comparaison_results.append({
        "strategy": name,
        "success_rate": success_rate,
        "avg_iter_success": avg_iter_success
    })
    
    print(f"| {name:<20} | Taux Succès: {success_rate:.1f}% | Iter. Moy.: {avg_iter_success:.0f}")

print("-" * 75)

# --- 3. VISUALISATION ---

strategies = [r['strategy'] for r in comparaison_results]
success = [r['success_rate'] for r in comparaison_results]
iters = [r['avg_iter_success'] for r in comparaison_results]

# Remplacer NaN par une valeur maximale pour le graphique si le taux de succès est 0
iters = [i if not np.isnan(i) else max_tot_iter for i in iters] 

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
x = np.arange(len(strategies))

# Graphique 1: Taux de Succès (Robustesse)
ax[0].bar(x, success, color=['skyblue', 'lightcoral', 'lightgreen'])
ax[0].set_xticks(x)
ax[0].set_xticklabels(strategies)
ax[0].set_ylabel('Taux de Succès (%)')
ax[0].set_title('Robustesse : Taux de Succès pour trouver la Source')
ax[0].set_ylim(0, 100)
for i, v in enumerate(success):
    ax[0].text(i, v + 2, f"{v:.1f}%", ha='center', color='black') # Affichage des valeurs

# Graphique 2: Itérations Moyennes (Efficacité)
ax[1].bar(x, iters, color=['skyblue', 'lightcoral', 'lightgreen'])
ax[1].set_xticks(x)
ax[1].set_xticklabels(strategies)
ax[1].set_ylabel("Itérations Moyennes (si succès)")
ax[1].set_title("Efficacité : Temps de Recherche Moyen")
for i, v in enumerate(iters):
    ax[1].text(i, v + 50, f"{v:.0f}", ha='center', color='black') # Affichage des valeurs

plt.tight_layout()
plt.show()