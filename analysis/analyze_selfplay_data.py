import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import base64
import html

# === Пътища ===
DEFAULT_DIR = Path(__file__).resolve().parent / "rl" / "selfplay_data"
ALT_DIR = Path("/home/presi/projects/chess_ai_project/training/training/rl/selfplay_data")
SELFPLAY_DIR = DEFAULT_DIR if DEFAULT_DIR.exists() else ALT_DIR
print(f"[INFO] Using self-play data from: {SELFPLAY_DIR}")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
LOG_DIR = Path(__file__).resolve().parent / "experiment_logs"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

for p in [PLOTS_DIR, LOG_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# === Зареждане на batch-ове ===
def load_batches(data_dir):
    batches = sorted(data_dir.glob("episode_*.pt"))
    print(f"Found {len(batches)} files\n")
    all_values, entropies, num_positions = [], [], []

    for i, fpath in enumerate(batches):
        data = torch.load(fpath, map_location="cpu")
        examples = data.get("examples", [])
        if len(examples) == 0:
            print(f"[WARN] {fpath.name} – empty batch.")
            continue

        vals, entrs = [], []
        for ex in examples:
            if len(ex) == 3:
                _, policy, value = ex
                vals.append(float(value) if not isinstance(value, torch.Tensor) else value.item())
                probs = policy.numpy() if isinstance(policy, torch.Tensor) else np.array(policy)
                ent = -np.sum(probs * np.log(probs + 1e-9))
                entrs.append(ent)

        if not vals:
            continue
        all_values.append(np.array(vals))
        entropies.append(np.mean(entrs) if entrs else np.nan)
        num_positions.append(len(examples))

    return all_values, entropies, num_positions


# === Графики и статистика ===
def plot_analysis(values_all, entropies, num_positions, save_path):
    if not values_all:
        print("[ERROR] No valid value data to plot.")
        return None, None

    avg_values_per_episode = [np.mean(v) for v in values_all]
    vals = np.concatenate(values_all)

    # win/draw/loss
    wins = np.sum(vals > 0.5)
    draws = np.sum(np.isclose(vals, 0.0, atol=1e-3))
    losses = np.sum(vals < 0.0)
    total = len(vals)
    win_p, draw_p, loss_p = (wins / total, draws / total, losses / total)

    print(f"\n[STATS]")
    print(f"  Total positions: {total}")
    print(f"  Win:  {wins} ({win_p:.1%})")
    print(f"  Draw: {draws} ({draw_p:.1%})")
    print(f"  Loss: {losses} ({loss_p:.1%})")

    # Корелация entropy ↔ value
    valid_pairs = [(e, v) for e, v in zip(entropies, avg_values_per_episode) if not np.isnan(e)]
    corr = np.corrcoef(*zip(*valid_pairs))[0, 1] if valid_pairs else np.nan
    print(f"  Correlation (Entropy ↔ Avg Value): {corr:.3f}\n")

    # === Плотове ===
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.hist(vals, bins=20, color="skyblue", edgecolor="black")
    plt.title("Value Target Distribution")
    plt.xlabel("Value target (win=1, draw=0, loss=-1)")
    plt.ylabel("Count")

    plt.subplot(3, 2, 2)
    plt.plot(entropies, marker="o")
    plt.title("Average Policy Entropy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")

    plt.subplot(3, 2, 3)
    plt.bar(range(len(num_positions)), num_positions)
    plt.title("Positions Collected per Episode")
    plt.xlabel("Episode")
    plt.ylabel("#Positions")

    plt.subplot(3, 2, 4)
    plt.plot(avg_values_per_episode, marker="o", color="green")
    plt.title("Average Value Target per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Mean value")

    plt.subplot(3, 1, 3)
    plt.scatter(entropies, avg_values_per_episode, color="purple", alpha=0.7)
    plt.title(f"Policy Entropy vs Avg Value (corr={corr:.2f})")
    plt.xlabel("Average Policy Entropy")
    plt.ylabel("Average Value Target")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot → {save_path}")

    return (win_p, draw_p, loss_p), corr


# === Логове на експерименти ===
def save_experiment_log(num_episodes, total_positions, win_p, draw_p, loss_p, corr, plot_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "total_positions": total_positions,
        "win_rate": win_p,
        "draw_rate": draw_p,
        "loss_rate": loss_p,
        "entropy_value_corr": corr,
        "plot_path": str(plot_path),
    }

    json_path = LOG_DIR / f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[LOG] Saved experiment → {json_path}")

    plot_experiment_trends(LOG_DIR)
    generate_html_report(REPORT_DIR, plot_path, LOG_DIR / "trend_summary.png", LOG_DIR)


def plot_experiment_trends(log_dir: Path):
    files = sorted(log_dir.glob("exp_*.json"))
    if not files:
        return
    data = []
    for f in files:
        with open(f) as j:
            data.append(json.load(j))
    ts = range(len(data))
    win = [d["win_rate"] for d in data]
    draw = [d["draw_rate"] for d in data]
    loss = [d["loss_rate"] for d in data]
    corr = [d["entropy_value_corr"] for d in data]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, win, label="Win", color="green")
    plt.plot(ts, draw, label="Draw", color="gray")
    plt.plot(ts, loss, label="Loss", color="red")
    plt.plot(ts, corr, label="Entropy–Value Corr", color="purple", linestyle="--")
    plt.legend()
    plt.xlabel("Experiment #")
    plt.ylabel("Rate / Corr")
    plt.title("Experiment Trends (Win/Draw/Loss + Entropy-Value Corr)")
    save_path = log_dir / "trend_summary.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Updated trend plot → {save_path}")


# === HTML отчет ===
def _img_to_base64(path: Path) -> str:
    try:
        b = path.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"
    except Exception:
        return ""


def generate_html_report(save_dir: Path, plot_path: Path, trend_path: Path, log_dir: Path):
    import html

    save_dir.mkdir(parents=True, exist_ok=True)
    json_logs = sorted(log_dir.glob("exp_*.json"))
    logs = []
    for j in json_logs:
        try:
            logs.append(json.load(open(j)))
        except Exception:
            continue

    latest = logs[-1] if logs else {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = save_dir / f"selfplay_report_{timestamp}.html"

    # Конвертираме изображенията в base64, за да се вградят директно
    def _img_to_base64(path):
        import base64
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

    plot_b64 = _img_to_base64(plot_path)
    trend_b64 = _img_to_base64(trend_path)

    # Безопасно форматиране
    win_txt  = f"{latest['win_rate']*100:.2f}%"  if isinstance(latest.get('win_rate'), (int, float)) else "-"
    draw_txt = f"{latest['draw_rate']*100:.2f}%" if isinstance(latest.get('draw_rate'), (int, float)) else "-"
    loss_txt = f"{latest['loss_rate']*100:.2f}%" if isinstance(latest.get('loss_rate'), (int, float)) else "-"
    corr_txt = f"{latest['entropy_value_corr']:.3f}" if isinstance(latest.get('entropy_value_corr'), (int, float)) else "-"

    # Таблица с цветове
    rows = []
    for rec in reversed(logs[-50:]):
        w = rec.get('win_rate', 0)
        d = rec.get('draw_rate', 0)
        l = rec.get('loss_rate', 0)
        c = rec.get('entropy_value_corr', 0)
        color_w = "style='color:#1b9400;font-weight:bold;'"  # зелено
        color_d = "style='color:#666;'"                      # сиво
        color_l = "style='color:#b30000;font-weight:bold;'"  # червено
        color_c = "" if abs(c) < 0.3 else ("style='color:#1b9400;'" if c < 0 else "style='color:#b30000;'")

        rows.append("<tr>" + "".join([
            f"<td>{html.escape(str(rec.get('timestamp','')))}</td>",
            f"<td>{rec.get('num_episodes','')}</td>",
            f"<td>{rec.get('total_positions','')}</td>",
            f"<td {color_w}>{w*100:.2f}%</td>",
            f"<td {color_d}>{d*100:.2f}%</td>",
            f"<td {color_l}>{l*100:.2f}%</td>",
            f"<td {color_c}>{c:.3f}</td>",
        ]) + "</tr>")

    table = "<table><thead><tr><th>Timestamp</th><th>Episodes</th><th>Positions</th><th>Win%</th><th>Draw%</th><th>Loss%</th><th>Corr</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"

    html_txt = f"""
    <html><head><meta charset='utf-8'><title>Selfplay Report</title>
    <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#fafafa;color:#222}}
    h2{{color:#333;}}
    .card{{background:#fff;padding:14px 18px;border-radius:10px;margin-bottom:16px;
           box-shadow:0 2px 5px rgba(0,0,0,0.1)}}
    table{{border-collapse:collapse;width:100%;font-size:14px;}}
    td,th{{border:1px solid #ddd;padding:8px;text-align:center}}
    th{{background:#f0f0f0}}
    img{{max-width:100%;border-radius:8px;margin-top:8px}}
    </style></head><body>
    <h2>♟ Self-play Training Report — {html.escape(timestamp)}</h2>

    <div class='card'><h3>Latest Summary</h3>
      <p><b>Episodes:</b> {latest.get('num_episodes','-')} &nbsp;|&nbsp;
         <b>Positions:</b> {latest.get('total_positions','-')}</p>
      <p><b>Win/Draw/Loss:</b> 
         <span style='color:#1b9400;font-weight:bold'>{win_txt}</span> /
         <span style='color:#666'>{draw_txt}</span> /
         <span style='color:#b30000;font-weight:bold'>{loss_txt}</span>
      </p>
      <p><b>Entropy–Value Corr:</b> {corr_txt}</p>
    </div>

    <div class='card'><h3>Charts</h3>
      {"<img src='"+plot_b64+"' alt='Summary plot'/>" if plot_b64 else "<p>No summary plot available.</p>"}
      {"<img src='"+trend_b64+"' alt='Trend plot'/>" if trend_b64 else ""}
    </div>

    <div class='card'><h3>Experiment History (last 50)</h3>{table}</div>

    </body></html>
    """
    out_file.write_text(html_txt, encoding="utf-8")
    print(f"[OK] HTML report → {out_file}")
    return out_file



# === Основен изпълним блок ===
if __name__ == "__main__":
    values_all, entropies, num_positions = load_batches(SELFPLAY_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = PLOTS_DIR / f"selfplay_summary_{timestamp}.png"

    results, corr = plot_analysis(values_all, entropies, num_positions, plot_path)
    if results:
        win_p, draw_p, loss_p = results
        save_experiment_log(len(values_all), sum(num_positions), win_p, draw_p, loss_p, corr, plot_path)