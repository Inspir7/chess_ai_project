import pandas as pd
import plotly.graph_objects as go
import time

CSV_PATH = "/home/presi/projects/chess_ai_project/training/logs/rl_stats.csv"
OUTPUT_HTML = "/home/presi/projects/chess_ai_project/analysis/rl_dashboard.html"


def preprocess_dataframe(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    run_ids = []
    current_run = 0

    for i in range(len(df)):
        if df.loc[i, "episode"] == 1:
            current_run += 1
        run_ids.append(current_run)

    df["run_id"] = run_ids

    df["global_episode"] = df.groupby("run_id").cumcount() + 1

    max_ep = df["episode"].max()
    df["global_episode"] = (df["run_id"] - 1) * max_ep + df["global_episode"]

    return df


def build_dashboard(df):

    # -----------------------
    # Exponential smoothing
    # -----------------------
    df["elo_smooth"] = df["elo_proxy"].ewm(span=30, adjust=False).mean()

    fig = go.Figure()

    # Avg Length
    fig.add_trace(go.Scatter(
        x=df["global_episode"],
        y=df["avg_len"],
        mode="lines",
        name="Avg Length",
        line=dict(width=2, color="#6c8ef5")
    ))

    # Draw Rate
    fig.add_trace(go.Scatter(
        x=df["global_episode"],
        y=df["draws"] / df["games"] * 100,
        mode="lines",
        name="Draw Rate (%)",
        line=dict(width=2, color="#ff6f61")
    ))

    # Elo proxy (raw)
    fig.add_trace(go.Scatter(
        x=df["global_episode"],
        y=df["elo_proxy"],
        mode="markers",
        name="Elo Proxy (raw)",
        marker=dict(size=4, opacity=0.4, color="#00f0b8")
    ))

    # Elo smoothed
    fig.add_trace(go.Scatter(
        x=df["global_episode"],
        y=df["elo_smooth"],
        mode="lines",
        name="Elo Proxy (smoothed)",
        line=dict(width=3, color="#00ffaa")
    ))

    # ------------------------
    # Markers for each RUN
    # ------------------------
    run_start_positions = df.groupby("run_id")["global_episode"].min()

    for run_id, x in run_start_positions.items():
        fig.add_vline(
            x=x,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Run {run_id}",
            annotation_position="top right"
        )

    fig.update_layout(
        title="AlphaZero RL Live Dashboard",
        xaxis_title="Global Episode Index",
        template="plotly_dark",
        height=650,
    )

    # Auto-refresh every 3 seconds
    html = fig.to_html(full_html=True)
    html = html.replace(
        "<body>",
        """<body>
        <script>
            setTimeout(function(){ location.reload(); }, 3000);
        </script>
        """
    )

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)


print("📊 Live dashboard is running…")

while True:
    try:
        df = pd.read_csv(CSV_PATH)

        # Filter out old test runs with 1–2 games
        df = df[df["games"] >= 20]

        if len(df) > 0:
            df = preprocess_dataframe(df)
            build_dashboard(df)

    except Exception as e:
        print("Error:", e)

    time.sleep(10)