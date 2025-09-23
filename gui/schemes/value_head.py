import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def draw_box(ax, label, description, xy, width=2.8, height=1.2, color='#ffd1dc'):
    box = FancyBboxPatch(
        (xy[0], xy[1]), width, height,
        boxstyle="round,pad=0.02",
        edgecolor='black',
        facecolor=color,
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2 + 0.15, label,
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(xy[0] + width / 2, xy[1] - 0.3, description,
            ha='center', va='top', fontsize=9, style='italic')


def draw_group_box(ax, x_start, x_end, label="", color='black', y=0, height=1.2):
    box = Rectangle((x_start - 0.2, y - 0.6), (x_end - x_start) + 3.0, height + 1.1,
                    linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--')
    ax.add_patch(box)
    ax.text((x_start + x_end) / 2 + 1.5, y + height + 0.7, label,
            ha='center', va='bottom', fontsize=9, color=color, weight='bold')


def plot_corrected_value_head():
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')

    layers = [
        ("Input\n[64×8×8]", "Изход от енкодера", '#ffd1dc'),
        ("1×1 Conv\n[1×8×8]", "Свиване на каналите", '#f9b3c2'),
        ("BatchNorm2d", "Нормализация", '#f4a7b9'),
        ("ReLU", "Нелинейност", '#f080a1'),
        ("Flatten\n[64]", "Оформяне към вектор", '#e96a8a'),
        ("Linear\n64→64", "Първи FC слой", '#e75480'),
        ("ReLU", "Активация", '#e04170'),
        ("Linear\n64→1", "Изход от регресия", '#de3163'),
        ("Tanh", "Ограничаване [-1,1]", '#d7265a'),
        ("Output value", "Оценка на позицията", '#cc1549')
    ]

    x = 0
    y = 0
    box_gap = 3.2
    positions = []

    for i, (label, desc, color) in enumerate(layers):
        draw_box(ax, label, desc, (x, y), color=color)
        positions.append(x)
        if i < len(layers) - 1:
            ax.annotate("",
                        xy=(x + 2.8, y + 0.6),
                        xytext=(x + 2.8 + 0.4, y + 0.6),
                        arrowprops=dict(arrowstyle="<-", lw=1.5))
        x += box_gap

    # Рамки с точен обхват
    draw_group_box(ax, positions[1], positions[3], label="Activation extraction", color='deeppink')
    draw_group_box(ax, positions[4], positions[8], label="Regression Evaluation", color='darkred')

    ax.set_xlim(-1, x + 1)
    ax.set_ylim(-1, 2.8)
    plt.tight_layout()
    plt.show()


plot_corrected_value_head()
