import pandas as pd
import plotly.graph_objects as go
import time

CSV_PATH = "/home/presi/projects/chess_ai_project/training/logs/rl_stats.csv"
OUTPUT_HTML = "/home/presi/projects/chess_ai_project/analysis/rl_dashboard.html"

def build_dashboard(df):
    fig = go.Figure()

    # Avg Length
    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["avg_len"],
        mode="lines+markers",
        name="Avg Length"
    ))

    # Draw Rate
    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["draws"] / df["games"] * 100,
        mode="lines+markers",
        name="Draw Rate (%)"
    ))

    # Elo Proxy
    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["elo_proxy"],
        mode="lines+markers",
        name="Elo Proxy"
    ))

    fig.update_layout(
        title="AlphaZero RL Live Dashboard",
        xaxis_title="Episode",
        template="plotly_dark",
        height=600
    )

    # Add auto-refresh every 3 seconds
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
print("Open in browser: file://" + OUTPUT_HTML)
print("Refreshing every 10 seconds…")

# Main loop
while True:
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) > 0:
            build_dashboard(df)
    except Exception as e:
        print("Error:", e)

    time.sleep(10)