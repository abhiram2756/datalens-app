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

def generate_static_charts(df):
    _base_style()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if "id" not in c.lower()]

    colors = _assign_colors(df.columns)
    chart_groups = []

    # BOXPLOT
    fig, ax = plt.subplots(figsize=(14,6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    data = [df[c].dropna() for c in numeric_cols]
    ax.boxplot(data)
    ax.set_xticklabels(numeric_cols, rotation=40)
    path = "static/boxplot.png"
    _save(fig, path)

    chart_groups.append({
        "label": "Box Plot",
        "icon": "📦",
        "scroll": "h",
        "charts": [{"title": "Box Plot", "path": path}]
    })

    # HEATMAP
    heat_cols = numeric_cols[:10]
    fig, ax = plt.subplots(figsize=(8,6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(df[heat_cols].corr(),
                annot=True, fmt=".2f",
                cmap="Oranges", ax=ax)
    path = "static/heatmap.png"
    _save(fig, path)

    chart_groups.append({
        "label": "Heatmap",
        "icon": "🌡️",
        "scroll": "h",
        "charts": [{"title": "Correlation Heatmap", "path": path}]
    })

    # TIME SERIES
    if "date" in df.columns and "price" in df.columns:
        df_ts = df.copy()
        df_ts["date"] = pd.to_datetime(df_ts["date"], errors="coerce")
        df_ts = df_ts.sort_values("date")

        fig, ax = plt.subplots(figsize=(12,5), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        ax.plot(df_ts["date"], df_ts["price"], color=colors["price"])
        ax.set_title("Trend Over Time – price", color=TEXT_COLOR)
        ax.grid(True)

        path = "static/timeseries.png"
        _save(fig, path)

        chart_groups.append({
            "label": "Time Series",
            "icon": "📈",
            "scroll": "h",
            "charts": [{"title": "Price over Time", "path": path}]
        })

    # 🔥 PIE CHART (ONLY NEW ADDITION)
    cat_cols = df.select_dtypes(include=["object"]).columns

    pie_cards = []
    for col in list(cat_cols)[:3]:
        counts = df[col].value_counts().head(6)

        fig, ax = plt.subplots(figsize=(6,6), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)

        ax.pie(counts,
               labels=counts.index,
               autopct="%1.1f%%")

        ax.set_title(f"{col} distribution", color=TEXT_COLOR)

        path = f"static/pie_{col}.png"
        _save(fig, path)

        pie_cards.append({
            "title": col,
            "path": path
        })

    if pie_cards:
        chart_groups.append({
            "label": "Pie Charts",
            "icon": "🥧",
            "scroll": "h",
            "charts": pie_cards
        })

    return chart_groups, colors


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

    charts, colors = generate_static_charts(df_num)

    table = df.head(10).to_html(classes="table table-bordered", index=False)
    stats = df_num.describe().round(2).to_html(classes="table table-bordered")

    return render_template('result.html',
                           table=table,
                           stats=stats,
                           chart_groups=charts,
                           summary={
                               "rows": df.shape[0],
                               "columns": df.shape[1],
                               "column_names": list(df.columns),
                               "numeric_columns": df_num.select_dtypes(include="number").columns.tolist(),
                               "missing_values": df.isnull().sum().to_dict()
                           },
                           col_colors=colors)


@app.route('/custom_scatter', methods=['POST'])
def custom_scatter():
    x = request.form.get("col_x")
    y = request.form.get("col_y")

    df = pd.read_csv("temp.csv").replace("N/A", np.nan)
    colors = _assign_colors(df.columns)

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    _scatter_single(ax, df, x, y, colors[x], colors[y])

    path = "static/custom_scatter.png"
    _save(fig, path)

    return render_template('result.html',
                           chart_groups=[{
                               "label": "Custom Scatter",
                               "icon": "⚙️",
                               "scroll": "h",
                               "charts":[{"title": f"{x} × {y}", "path": path}]
                           }])


@app.route('/custom_hist', methods=['POST'])
def custom_hist():
    col = request.form.get("col")

    df = pd.read_csv("temp.csv").replace("N/A", np.nan)
    colors = _assign_colors(df.columns)

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    ax.hist(df[col].dropna(), color=colors[col], bins=25)
    ax.set_title(f"Distribution – {col}", color=TEXT_COLOR)
    ax.grid(True)

    path = "static/custom_hist.png"
    _save(fig, path)

    return render_template('result.html',
                           chart_groups=[{
                               "label": "Custom Histogram",
                               "icon": "⚙️",
                               "scroll": "h",
                               "charts":[{"title": col, "path": path}]
                           }])


if __name__ == '__main__':
    app.run(debug=True)
