from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

BG_COLOR   = "#0A0A0A"
CARD_BG    = "#141414"
TEXT_COLOR = "#F0F0F0"
GRID_COLOR = "#2A2A2A"

COLOR_POOL = [
    "#FF6B6B","#FF9F43","#FECA57","#48DBFB","#1DD1A1",
    "#54A0FF","#5F27CD","#FF9FF3","#00D2D3","#FF6348",
    "#2ECC71","#3498DB","#9B59B6","#F39C12","#E74C3C",
    "#1ABC9C","#E67E22","#16A085","#8E44AD","#2980B9",
    "#F1C40F","#D35400","#27AE60","#C0392B","#7F8C8D",
    "#A29BFE","#FD79A8","#FDCB6E","#6C5CE7","#00CEC9"
]

def _assign_colors(columns):
    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    return {col: pool[i % len(pool)] for i, col in enumerate(columns)}

def _base_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_BG,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": "#AAAAAA",
        "ytick.color": "#AAAAAA",
        "grid.color": GRID_COLOR,
    })

def _save(fig, path):
    os.makedirs("static", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

def _scatter_single(ax, df, x, y, cx, cy):
    valid = df[[x, y]].dropna()
    ax.scatter(valid[x], valid[y], alpha=0.65, color=cx, s=30)
    ax.set_title(f"{x} × {y}", color=TEXT_COLOR)
    ax.set_xlabel(x, color=cx)
    ax.set_ylabel(y, color=cy)
    ax.grid(True)

# ================= CHARTS =================

def generate_charts(df):
    _base_style()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if "id" not in c.lower()]
    cat_cols = df.select_dtypes(include=["object"]).columns

    colors = _assign_colors(df.columns)
    chart_groups = []

    # BOXPLOT
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(14,6), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        data = [df[c].dropna() for c in numeric_cols]
        ax.boxplot(data)
        ax.set_xticklabels(numeric_cols, rotation=40)
        _save(fig, "static/boxplot.png")

        chart_groups.append({
            "label": "Box Plot",
            "icon": "📦",
            "charts": [{"title": "Box Plot", "path": "static/boxplot.png"}]
        })

    # 🔥 HEATMAP (ALL COLUMNS SAFE)
    if len(numeric_cols) >= 2:
        n = len(numeric_cols)
        size = max(8, n * 0.7)

        fig, ax = plt.subplots(figsize=(size, size), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)

        annot_flag = n <= 15  # prevent clutter

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=annot_flag,
            fmt=".2f",
            cmap="Oranges",
            ax=ax
        )

        _save(fig, "static/heatmap.png")

        chart_groups.append({
            "label": "Heatmap",
            "icon": "🌡️",
            "charts": [{"title": "Correlation Heatmap", "path": "static/heatmap.png"}]
        })

    # PIE
    pie_cards = []
    for col in list(cat_cols)[:3]:
        series = df[col]

        if "date" in col.lower():
            series = pd.to_datetime(series, errors="coerce")
            series = series.dt.to_period("M").astype(str)

        counts = series.value_counts().head(6)

        fig, ax = plt.subplots(figsize=(6,6), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)

        wedges, _, _ = ax.pie(counts, autopct="%1.1f%%")

        ax.legend(wedges, counts.index,
                  title=col,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=8)

        ax.set_title(f"{col} distribution", color=TEXT_COLOR)

        path = f"static/pie_{col}.png"
        _save(fig, path)

        pie_cards.append({"title": col, "path": path})

    if pie_cards:
        chart_groups.append({
            "label": "Pie Charts",
            "icon": "🥧",
            "charts": pie_cards
        })

    # 🔥 TIME SERIES (OLD DESIGN RESTORED)
    if "date" in df.columns and numeric_cols:
        val = numeric_cols[0]

        df_ts = df.copy()
        df_ts["date"] = pd.to_datetime(df_ts["date"], errors="coerce")
        df_ts = df_ts.sort_values("date")

        fig, ax = plt.subplots(figsize=(12,5), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)

        ax.plot(df_ts["date"], df_ts[val], color=colors[val])

        ax.set_title(f"Trend Over Time – {val}", color=TEXT_COLOR)
        ax.set_xlabel("Date", color=TEXT_COLOR)
        ax.set_ylabel(val, color=colors[val])

        ax.grid(True)

        _save(fig, "static/timeseries.png")

        chart_groups.append({
            "label": "Time Series",
            "icon": "📈",
            "charts": [{"title": f"{val} over Time", "path": "static/timeseries.png"}]
        })

    return chart_groups, colors

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')

    df = pd.read_csv(file, nrows=2000)
    df = df.fillna("N/A")

    df.to_csv("temp.csv", index=False)

    df_num = df.replace("N/A", np.nan)

    charts, colors = generate_charts(df_num)

    return render_template('result.html',
        table=df.head(10).to_html(index=False),
        stats=df_num.describe().to_html(),
        chart_groups=charts,
        summary={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "numeric_columns": df_num.select_dtypes(include="number").columns.tolist(),
            "missing_values": df.isnull().sum().to_dict()
        },
        col_colors=colors
    )

# ================= CUSTOM =================

@app.route('/custom_scatter', methods=['POST'])
def custom_scatter():
    df = pd.read_csv("temp.csv")
    df_num = df.replace("N/A", np.nan)

    x = request.form.get("col_x")
    y = request.form.get("col_y")

    colors = _assign_colors(df.columns)

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    _scatter_single(ax, df_num, x, y, colors[x], colors[y])

    path = "static/custom_scatter.png"
    _save(fig, path)

    return render_template('result.html',
        chart_groups=[{
            "label":"Custom Scatter",
            "charts":[{"title":f"{x} × {y}","path":path}]
        }],
        summary={
            "column_names": list(df.columns),
            "numeric_columns": list(df_num.select_dtypes(include="number").columns)
        },
        table=df.head().to_html(),
        stats=df_num.describe().to_html(),
        col_colors=colors
    )

@app.route('/custom_hist', methods=['POST'])
def custom_hist():
    df = pd.read_csv("temp.csv")
    df_num = df.replace("N/A", np.nan)

    col = request.form.get("col")

    colors = _assign_colors(df.columns)

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    ax.hist(df_num[col].dropna(), color=colors[col])
    ax.set_title(f"Distribution – {col}", color=TEXT_COLOR)
    ax.grid(True)

    path = "static/custom_hist.png"
    _save(fig, path)

    return render_template('result.html',
        chart_groups=[{
            "label":"Custom Histogram",
            "charts":[{"title":col,"path":path}]
        }],
        summary={
            "column_names": list(df.columns),
            "numeric_columns": list(df_num.select_dtypes(include="number").columns)
        },
        table=df.head().to_html(),
        stats=df_num.describe().to_html(),
        col_colors=colors
    )

if __name__ == '__main__':
    app.run(debug=True)
