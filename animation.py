import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CHANGEMENT ICI : On importe depuis strategies3 ---
from strategies3 import strategy_simple, strategy_spiral, strategy_mosquito

# --- 1. CONFIGURATION ---
STRATEGY_NAME = 'mosquito'  # Choix : 'simple', 'spiral' ou 'mosquito'

# Paramètres de simulation
domain_x, domain_y = 70, 50
a, b = 6, 4
source_x, source_y = 2, (domain_y-b)//2
max_tot_iter = 3000
start_x, start_y = domain_x-1, np.random.randint(0, domain_y)

# Paramètres spécifiques aux stratégies
d_simple = 4        # Pour la stratégie simple (distance de remontée)
aug_ampl = 1.5      # Pour mosquito (facteur d'augmentation du casting)

# --- 2. GÉNÉRATION DU CHAMP ---
print("--- Démarrage de la simulation ---")
print("1. Génération du champ d'odeur...")
V = 2.0; D = 1.0; tau = 10
X, Y = np.meshgrid(np.arange(domain_x), np.arange(domain_y))
c = np.zeros_like(X, dtype=float)
downwind = X > source_x
xdist = X - source_x
ydist = Y - source_y
with np.errstate(divide='ignore', invalid='ignore'):
    spread = 4 * D * xdist / V
    in_field = downwind & (spread > 0)
    c[in_field] = np.exp(-xdist[in_field] / (V * tau)) * np.exp(-ydist[in_field]**2 / (2 * spread[in_field]))

np.random.seed()
concentration = (np.random.rand(*c.shape) < (c / c.max())).astype(int)
concentration[source_y:source_y+b, source_x:source_x+a] = 1
y_pts, x_pts = np.where(concentration == 1)

# --- 3. EXÉCUTION DE LA STRATÉGIE (Mise à jour pour strategies3) ---
print(f"2. Exécution de la stratégie : {STRATEGY_NAME}...")

if STRATEGY_NAME == 'simple':
    # strategies3 demande : (..., max_tot_iter, d)
    found, trajet, _ = strategy_simple(concentration, source_x, source_y, a, b, 
                                       start_x, start_y, max_tot_iter, d=d_simple)

elif STRATEGY_NAME == 'spiral':
    # strategies3 demande : (..., T_loss, max_tot_iter)
    found, trajet, _ = strategy_spiral(concentration, source_x, source_y, a, b, 
                                       start_x, start_y, max_tot_iter=max_tot_iter)

elif STRATEGY_NAME == 'mosquito':
    # strategies3 demande : (..., max_tot_iter, aug_ampl)
    found, trajet, _ = strategy_mosquito(concentration, source_x, source_y, a, b, 
                                         start_x, start_y, max_tot_iter, aug_ampl=aug_ampl)
else:
    raise ValueError("Stratégie inconnue")

print(f"   -> Trajet terminé en {len(trajet)} étapes.")

# --- FONCTION DE CONFIGURATION GRAPHIQUE ---
def create_animation_object(fig, ax):
    ax.set_xlim(0, domain_x)
    ax.set_ylim(0, domain_y)
    ax.set_title(f'Simulation: {STRATEGY_NAME.capitalize()} Strategy')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Fond (Odeur + Source)
    ax.scatter(x_pts, y_pts, c='blue', s=2, alpha=0.3, label='Odeur')
    rect = plt.Rectangle((source_x, source_y), a, b, edgecolor='r', facecolor='none', lw=2, label='Source')
    ax.add_patch(rect)

    # Éléments mobiles
    line, = ax.plot([], [], 'r-', alpha=0.6, lw=1, label='Trajet')
    point, = ax.plot([], [], 'ro', ms=6, label='Moustique') 

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        current_pos = trajet[frame]
        path_x, path_y = zip(*trajet[:frame+1])
        line.set_data(path_x, path_y)
        point.set_data([current_pos[0]], [current_pos[1]])
        return line, point

    return FuncAnimation(fig, update, frames=len(trajet), init_func=init, blit=True, interval=40)


# --- 4. AFFICHAGE À L'ÉCRAN ---
print("3. Affichage de l'animation... (Fermez la fenêtre pour continuer)")
fig_screen, ax_screen = plt.subplots(figsize=(10, 6))
ani_screen = create_animation_object(fig_screen, ax_screen)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show() 


# --- 5. DEMANDE DE SAUVEGARDE ---
print("-" * 30)
reponse = input("Voulez-vous sauvegarder cette simulation en vidéo MP4 ? (o/n) : ")

if reponse.lower().startswith('o') or reponse.lower() == 'y':
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_filename = f'simulation_{STRATEGY_NAME}.mp4'
    full_output_path = os.path.join(desktop_path, output_filename)
    
    print(f"   Préparation de la sauvegarde vers : {output_filename}")
    print("   Veuillez patienter pendant l'encodage...")

    fig_save, ax_save = plt.subplots(figsize=(10, 6))
    ani_save = create_animation_object(fig_save, ax_save)
    ax_save.legend(loc='upper right')
    plt.tight_layout()

    try:
        ani_save.save(full_output_path, writer='ffmpeg', fps=30, dpi=150)
        print(f"✅ SUCCÈS ! Vidéo sauvegardée sur le Bureau.")
    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde : {e}")
        
    plt.close(fig_save)

else:
    print("❌ Pas de sauvegarde. Fin du programme.")