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
        "figure.facecolor":   BG_COLOR,
        "axes.facecolor":     CARD_BG,
        "axes.edgecolor":     "#2A2A2A",
        "axes.labelcolor":    TEXT_COLOR,
        "xtick.color":        "#AAAAAA",
        "ytick.color":        "#AAAAAA",
        "text.color":         TEXT_COLOR,
        "grid.color":         GRID_COLOR,
        "grid.linestyle":     "--",
        "grid.alpha":         0.4,
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "figure.titlesize":   16,
        "figure.titleweight": "bold",
    })


def _save(fig, path):
    os.makedirs("static", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _safe_corr(df, cols):
    corr_df = df[cols].corr().abs()
    arr = np.array(corr_df.values, dtype=float, copy=True)
    return corr_df, arr


def _histogram(ax, series, color):
    ax.hist(series.dropna(), bins=25, color=color, alpha=0.85, edgecolor="#0A0A0A")
    ax.set_title(f"Distribution – {series.name}", color=TEXT_COLOR, pad=12)
    ax.set_xlabel(series.name, labelpad=8)
    ax.set_ylabel("Frequency", labelpad=8)
    ax.grid(True, axis="y")
    mean = series.mean()
    ax.axvline(mean, color="#FFFFFF", lw=2, linestyle="--",
               label=f"Mean: {mean:.2f}")
    ax.legend(fontsize=10)


def _scatter_single(ax, df, col_x, col_y, color_x, color_y):
    valid = df[[col_x, col_y]].dropna()
    ax.scatter(valid[col_x], valid[col_y],
               alpha=0.65, color=color_x, s=30, edgecolors="none")
    ax.set_title(f"{col_x}  ×  {col_y}", color=TEXT_COLOR, pad=12)
    ax.set_xlabel(col_x, color=color_x, labelpad=8, fontsize=12)
    ax.set_ylabel(col_y, color=color_y, labelpad=8, fontsize=12)
    ax.tick_params(axis="x", colors=color_x)
    ax.tick_params(axis="y", colors=color_y)
    ax.grid(True)


def generate_charts(df):
    _base_style()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if "id" not in c.lower() and "index" not in c.lower()]

    col_colors = _assign_colors(list(df.columns))
    chart_groups = []

    if len(numeric_cols) >= 2:
        scatter_cards = []
        for gi, (cx, cy) in enumerate(combinations(numeric_cols, 2)):
            fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
            ax.set_facecolor(CARD_BG)
            _scatter_single(ax, df, cx, cy, col_colors[cx], col_colors[cy])
            plt.tight_layout(pad=1.5)
            path = f"static/chart_scatter_{gi}.png"
            _save(fig, path)
            scatter_cards.append({
                "title": f"Scatter – {cx} × {cy}",
                "hint": "",
                "path": path,
                "colors": {cx: col_colors[cx], cy: col_colors[cy]}
            })

        chart_groups.append({
            "label": "Scatter Plots",
            "icon": "🔵",
            "scroll": "v",
            "charts": scatter_cards
        })

    return chart_groups, col_colors


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('index.html')

    file = request.files.get('file')
    try:
        chunk = pd.read_csv(file, chunksize=5000)
        df = next(chunk)
        df = df.drop_duplicates()
        df = df.fillna("N/A")

        # ✅ REQUIRED FIX: SAVE DATA
        df.to_csv("temp.csv", index=False)

    except Exception as e:
        return f"Error reading file: {e}"

    df_num = df.replace("N/A", np.nan)

    chart_groups, col_colors = generate_charts(df_num)

    return render_template('result.html',
                           chart_groups=chart_groups,
                           summary={
                               "rows": df.shape[0],
                               "columns": df.shape[1],
                               "column_names": list(df.columns),
                               "numeric_columns": df_num.select_dtypes(include="number").columns.tolist(),
                               "missing_values": df.isnull().sum().to_dict(),
                           },
                           col_colors=col_colors)


# ✅ REQUIRED FIX: CUSTOM ROUTE
@app.route('/custom', methods=['POST'])
def custom_plot():
    col_x = request.form.get('col_x')
    col_y = request.form.get('col_y')

    df = pd.read_csv("temp.csv")
    df = df.replace("N/A", np.nan)

    col_colors = _assign_colors(list(df.columns))

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    _scatter_single(ax, df, col_x, col_y,
                    col_colors.get(col_x, "#FFFFFF"),
                    col_colors.get(col_y, "#FFFFFF"))

    path = "static/custom_plot.png"
    _save(fig, path)

    chart_groups = [{
        "label": "Custom Plot",
        "icon": "⚙️",
        "scroll": "h",
        "charts": [{
            "title": f"{col_x} × {col_y}",
            "hint": "User selected columns",
            "path": path,
            "colors": {
                col_x: col_colors.get(col_x),
                col_y: col_colors.get(col_y)
            }
        }]
    }]

    return render_template('result.html',
                           chart_groups=chart_groups,
                           summary={
                               "rows": df.shape[0],
                               "columns": df.shape[1],
                               "column_names": list(df.columns),
                               "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                               "missing_values": df.isnull().sum().to_dict(),
                           },
                           col_colors=col_colors)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
