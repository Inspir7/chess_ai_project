import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, label, description, xy, width=2.8, height=1.2, color='#ffc0cb'):
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


def plot_policy_head_architecture():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    layers = [
        ("Input\n[64×8×8]", "\nИзход от енкодера", '#ffd1dc'),
        ("1×1 Conv\n[2×8×8]", "Намалява каналите до 2", '#f9b3c2'),
        ("BatchNorm2d\n[2×8×8]", "\nНормализация на активациите", '#f4a7b9'),
        ("ReLU\n[2×8×8]", "Нелинейна трансформация", '#f080a1'),
        ("Flatten\n[128]", "\nОформяне за линейния слой", '#e96a8a'),
        ("Linear\n[4672]", "Генериране на логити за ходове", '#e75480'),
        ("Output logits", "\nНай-добрите следващи ходове", '#de3163')
    ]

    x = 0
    y = 0
    box_gap = 3.2

    for i, (label, desc, color) in enumerate(layers):
        draw_box(ax, label, desc, (x, y), color=color)
        if i < len(layers) - 1:
            ax.annotate("",
                        xy=(x + 2.8, y + 0.6),
                        xytext=(x + 2.8 + 0.4, y + 0.6),
                        arrowprops=dict(arrowstyle="<-", lw=1.5))
        x += box_gap

    # Групиране: Feature Transformation
    ax.add_patch(FancyBboxPatch(
        (-0.2, y - 0.6),        # Начална точка
        3 * box_gap + 3.1,      # ширина
        2.1,                    # височина
        boxstyle="round,pad=0.02",
        edgecolor='deeppink',
        facecolor='none',
        linewidth=2,
        linestyle='--'
    ))
    ax.text(2.5, y + 1.6, "Feature Transformation", fontsize=10, color='deeppink', ha='center')

    # Групиране: Classification
    ax.add_patch(FancyBboxPatch(
        (3 * box_gap + 3.1, y - 0.6),
        3 * box_gap,
        2.1,
        boxstyle="round,pad=0.02",
        edgecolor='darkred',
        facecolor='none',
        linewidth=2,
        linestyle='--'
    ))
    ax.text(3 * box_gap + 4.7, y + 1.6, "Classification", fontsize=10, color='darkred', ha='center')

    ax.set_xlim(-1, x + 1)
    ax.set_ylim(-1, 2.5)
    plt.tight_layout()
    plt.show()


plot_policy_head_architecture()
