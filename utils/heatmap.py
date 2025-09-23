import matplotlib.pyplot as plt
import numpy as np

def plot_feature_map(heatmap, title, cmap='hot'):
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(heatmap, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
    ax.set_yticklabels([str(i) for i in range(1, 9)])
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Интензитет на заплахата')
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()

# Кон
knight_map = np.zeros((8, 8))
knight_map[0, 1] = 0.8
knight_map[2, 6] = 0.6
knight_map[5, 4] = 1.0

# Топ
rook_map = np.zeros((8, 8))
rook_map[4, :] = 0.4
rook_map[:, 3] = 0.4
rook_map[4, 3] = 1.0

# Офицер
bishop_map = np.zeros((8, 8))
for i in range(8):
    if 0 <= 2+i < 8 and 0 <= 2+i < 8:
        bishop_map[2+i, 2+i] = 0.6
    if 0 <= 5-i < 8 and 0 <= 2+i < 8:
        bishop_map[5-i, 2+i] = 0.6
bishop_map[2, 2] = 1.0

# Комбинирана карта
combined_map = (knight_map + rook_map + bishop_map) / 3

# Визуализации
plot_feature_map(knight_map, "Характеристична карта – заплахи от Кон", cmap='Reds')
plot_feature_map(rook_map, "Характеристична карта – заплахи от Топ", cmap='Blues')
plot_feature_map(bishop_map, "Характеристична карта – заплахи от Офицер", cmap='Greens')
plot_feature_map(combined_map, "Комбинирана характеристична карта", cmap='magma')
