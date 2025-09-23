import matplotlib.pyplot as plt
import numpy as np

# Размер на дъската
board_size = 8

# Координати на фигурите
pieces = {
    'knight': (4, 4),   # e5
    'rook': (6, 1),     # g2
    'bishop': (2, 6),   # c7
}

# Функции за заплахи
def knight_threat(x, y):
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    threat = np.zeros((board_size, board_size))
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            threat[ny, nx] = 1.0
    return threat

def rook_threat(x, y):
    threat = np.zeros((board_size, board_size))
    for i in range(board_size):
        if i != x:
            threat[y, i] = 0.8
        if i != y:
            threat[i, x] = 0.8
    threat[y, x] = 1.0
    return threat

def bishop_threat(x, y):
    threat = np.zeros((board_size, board_size))
    for d in range(1, board_size):
        for dx, dy in [(-d, -d), (-d, d), (d, -d), (d, d)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                threat[ny, nx] = 0.6
    threat[y, x] = 1.0
    return threat

# Характеристични карти
knight_map = knight_threat(*pieces['knight'])
rook_map = rook_threat(*pieces['rook'])
bishop_map = bishop_threat(*pieces['bishop'])
combined_map = (knight_map + rook_map + bishop_map) / 3.0

# Визуализация
def plot_threat_map(data, title, cmap):
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
    ax.set_yticklabels([str(i) for i in range(1, 9)])
    ax.set_title(title)
    plt.colorbar(cax, label='Интензитет на заплахата', shrink=0.8)
    plt.grid(False)
    plt.tight_layout()
    return fig

plot_threat_map(knight_map, "Заплахи от Кон (на e5)", 'Reds')
plot_threat_map(rook_map, "Заплахи от Топ (на g2)", 'Blues')
plot_threat_map(bishop_map, "Заплахи от Офицер (на c7)", 'Greens')
plot_threat_map(combined_map, "Комбинирана карта на заплахи", 'magma')

plt.show()
