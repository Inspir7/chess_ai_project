import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, label, desc, xy, width=2.8, height=1.2, color='#ffc0cb'):
    box = FancyBboxPatch((xy[0], xy[1]), width, height,
                         boxstyle="round,pad=0.02", edgecolor='black',
                         facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2 + 0.15,
            label, ha='center', va='center', fontsize=10, weight='bold')
    ax.text(xy[0] + width / 2, xy[1] - 0.3, desc,
            ha='center', va='top', fontsize=9, style='italic')


def plot_full_alphazero_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    x_start = 0
    y_center = 0
    gap_x = 3.4

    draw_box(ax, "Input\n[15×8×8]", "FEN → Tensor", (x_start, y_center))

    encoder_x = x_start + gap_x
    for i in range(3):
        draw_box(
            ax,
            f"ConvBlock {i+1}\n[64×8×8]",
            "Conv + BN + ReLU",
            (encoder_x + i * gap_x, y_center),
            color='#f9b3c2'
        )

    encoder_out_x = encoder_x + 3 * gap_x

    # Policy Head
    policy_y = y_center + 3
    draw_box(ax, "1×1 Conv\n[2×8×8]", "Свиване на каналите", (encoder_out_x, policy_y), color='#f7a1b0')
    draw_box(ax, "Flatten\n[128]", "Оформяне на вектор", (encoder_out_x + gap_x, policy_y), color='#f080a1')
    draw_box(ax, "Linear\n[128→4672]", "Логити за ходове", (encoder_out_x + 2 * gap_x, policy_y), color='#e75480')
    draw_box(ax, "Output\n[4672]", "Вероятност за ход", (encoder_out_x + 3 * gap_x, policy_y), color='#de3163')

    # Value Head
    value_y = y_center - 3
    draw_box(ax, "1×1 Conv\n[1×8×8]", "Свиване на каналите", (encoder_out_x, value_y), color='#f7a1b0')
    draw_box(ax, "Flatten\n[64]", "Оформяне на вектор", (encoder_out_x + gap_x, value_y), color='#f080a1')
    draw_box(ax, "Linear\n[64→64]", "FC слой", (encoder_out_x + 2 * gap_x, value_y), color='#e75480')
    draw_box(ax, "Linear\n[64→1]", "Изход от регресия", (encoder_out_x + 3 * gap_x, value_y), color='#de3163')
    draw_box(ax, "Tanh\n[-1, 1]", "Ограничаване", (encoder_out_x + 4 * gap_x, value_y), color='#c71585')
    draw_box(ax, "Output\n[1]", "Оценка на позицията", (encoder_out_x + 5 * gap_x, value_y), color='#c2185b')

    # Стрелки
    def arrow(from_xy, to_xy):
        ax.annotate("",
                    xy=to_xy,
                    xytext=from_xy,
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow((x_start + 2.8, y_center + 0.6), (encoder_x, y_center + 0.6))
    arrow((encoder_out_x - 0.6, y_center + 0.6), (encoder_out_x, policy_y + 0.6))
    arrow((encoder_out_x - 0.6, y_center + 0.6), (encoder_out_x, value_y + 0.6))

    ax.set_xlim(-1, encoder_out_x + 6 * gap_x + 1)
    ax.set_ylim(value_y - 1.5, policy_y + 2)
    plt.tight_layout()
    plt.show()


plot_full_alphazero_architecture()
