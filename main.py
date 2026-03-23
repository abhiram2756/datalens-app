from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import random
import os
from itertools import combinations

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # increased

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
    "#A29BFE","#FD79A8","#FDCB6E","#6C5CE7","#00CEC9",
    "#E17055","#74B9FF","#55EFC4","#B2BEC3","#FAB1A0",
    "#81ECEC","#DFE6E9","#FD63D0","#45AAF2","#26DE81",
    "#FC5C65","#FD9644","#F7B731","#20BF6B","#0FB9B1",
    "#2BCBBA","#778CA3","#A55EEA","#4B7BEC","#D1D8E0",
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
    })

def _save(fig, path):
    os.makedirs("static", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def generate_charts(df):
    _base_style()

    # ✅ LIMIT TO FIRST 5 NUMERIC COLUMNS
    numeric_cols = [c for c in df.select_dtypes(include="number").columns][:5]
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns][:5]

    col_colors = _assign_colors(df.columns)
    chart_groups = []

    # HISTOGRAMS (LIMITED)
    hist_cards = []
    for i, col in enumerate(numeric_cols[:10]):
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna())
        path = f"static/hist_{i}.png"
        _save(fig, path)
        hist_cards.append({"title": col, "path": path, "colors": {col: col_colors[col]}})
    if hist_cards:
        chart_groups.append({"label": "Histograms", "icon": "📊", "scroll": "v", "charts": hist_cards})

    # SCATTER (LIMITED)
    pairs = list(combinations(numeric_cols, 2))[:10]
    scatter_cards = []
    for i, (x, y) in enumerate(pairs):
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y])
        path = f"static/scatter_{i}.png"
        _save(fig, path)
        scatter_cards.append({"title": f"{x} vs {y}", "path": path, "colors": {}})
    if scatter_cards:
        chart_groups.append({"label": "Scatter", "icon": "🔵", "scroll": "v", "charts": scatter_cards})

    return chart_groups, col_colors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    # ✅ REDUCE SIZE
    df = pd.read_csv(file, nrows=2000)
    df = df.fillna("N/A")

    # ✅ SAVE FOR CUSTOM
    df.to_csv("temp.csv", index=False)

    df_num = df.replace("N/A", np.nan)

    charts, colors = generate_charts(df_num)

    return render_template('result.html',
                           table=df.head().to_html(),
                           stats=df_num.describe().to_html(),
                           chart_groups=charts,
                           summary={
                               "rows": df.shape[0],
                               "columns": df.shape[1],
                               "column_names": list(df.columns),
                               "numeric_columns": df_num.select_dtypes(include="number").columns.tolist(),
                               "missing_values": df.isnull().sum().to_dict()
                           },
                           col_colors=colors)

# ✅ CUSTOM PLOT
@app.route('/custom', methods=['POST'])
def custom():
    col_x = request.form.get("col_x")
    col_y = request.form.get("col_y")

    df = pd.read_csv("temp.csv")

    fig, ax = plt.subplots()
    ax.scatter(df[col_x], df[col_y])

    path = "static/custom.png"
    fig.savefig(path)
    plt.close(fig)

    return render_template('result.html',
                           chart_groups=[{
                               "label": "Custom",
                               "icon": "⚙️",
                               "scroll": "h",
                               "charts": [{"title": "Custom Plot", "path": path, "colors": {}}]
                           }])
