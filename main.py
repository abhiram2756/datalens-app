from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from itertools import combinations

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

def _scatter_single(ax, df, col_x, col_y, color_x, color_y):
    valid = df[[col_x, col_y]].dropna()
    ax.scatter(valid[col_x], valid[col_y],
               alpha=0.65, color=color_x, s=30, edgecolors="none")
    ax.set_title(f"{col_x}  ×  {col_y}", color=TEXT_COLOR)
    ax.set_xlabel(col_x, color=color_x)
    ax.set_ylabel(col_y, color=color_y)
    ax.tick_params(axis="x", colors=color_x)
    ax.tick_params(axis="y", colors=color_y)
    ax.grid(True)

def generate_charts(df):
    _base_style()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns]
    limited_cols = numeric_cols[:10]

    col_colors = _assign_colors(df.columns)
    chart_groups = []

    # Boxplot
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(12,6), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        data = [df[c].dropna() for c in limited_cols]
        ax.boxplot(data)
        ax.set_xticklabels(limited_cols, rotation=40)
        path = "static/boxplot.png"
        _save(fig, path)

        chart_groups.append({
            "label": "Box Plot",
            "icon": "📦",
            "scroll": "h",
            "charts": [{
                "title": "Box Plot",
                "hint": "",
                "path": path,
                "colors": {c: col_colors[c] for c in limited_cols}
            }]
        })

    # Heatmap
    if len(limited_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8,6), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        sns.heatmap(df[limited_cols].corr(), ax=ax)
        path = "static/heatmap.png"
        _save(fig, path)

        chart_groups.append({
            "label": "Heatmap",
            "icon": "🌡️",
            "scroll": "h",
            "charts": [{
                "title": "Correlation Heatmap",
                "hint": "",
                "path": path,
                "colors": {c: col_colors[c] for c in limited_cols}
            }]
        })

    # Scatter
    pairs = list(combinations(limited_cols, 2))[:10]
    scatter_cards = []
    for i,(x,y) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        _scatter_single(ax, df, x, y, col_colors[x], col_colors[y])
        path = f"static/scatter_{i}.png"
        _save(fig, path)
        scatter_cards.append({
            "title": f"{x} × {y}",
            "hint": "",
            "path": path,
            "colors": {x: col_colors[x], y: col_colors[y]}
        })

    if scatter_cards:
        chart_groups.append({
            "label": "Scatter",
            "icon": "🔵",
            "scroll": "v",
            "charts": scatter_cards
        })

    return chart_groups, col_colors


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

    # ✅ RESTORED PART
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


@app.route('/custom', methods=['POST'])
def custom():
    col_x = request.form.get("col_x")
    col_y = request.form.get("col_y")

    df = pd.read_csv("temp.csv").replace("N/A", np.nan)
    colors = _assign_colors(df.columns)

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    _scatter_single(ax, df, col_x, col_y, colors[col_x], colors[col_y])

    path = "static/custom.png"
    _save(fig, path)

    return render_template('result.html',
                           chart_groups=[{
                               "label": "Custom",
                               "icon": "⚙️",
                               "scroll": "h",
                               "charts":[{
                                   "title": f"{col_x} × {col_y}",
                                   "hint": "",
                                   "path": path,
                                   "colors": {col_x: colors[col_x], col_y: colors[col_y]}
                               }]
                           }],
                           summary={
                               "rows": df.shape[0],
                               "columns": df.shape[1],
                               "column_names": list(df.columns),
                               "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                               "missing_values": df.isnull().sum().to_dict()
                           },
                           col_colors=colors)


if __name__ == '__main__':
    app.run(debug=True)
