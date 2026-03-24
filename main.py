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


# ─────────────── chart builders ─────────────────────────────────────────────

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


def _boxplot(ax, df, cols, col_colors):
    data = [df[c].dropna().values for c in cols]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    widths=0.55,
                    medianprops=dict(color="#FFFFFF", linewidth=3.5),
                    whiskerprops=dict(color="#AAAAAA", linewidth=2.0),
                    capprops=dict(color="#AAAAAA", linewidth=2.5),
                    boxprops=dict(linewidth=2.0),
                    flierprops=dict(marker='o', color='#AAAAAA',
                                   alpha=0.4, markersize=5))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col_colors[col])
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=13)
    ax.set_title("Box Plot – All Numeric Columns", color=TEXT_COLOR, pad=20, fontsize=17)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", linewidth=0.8)

    # Auto log scale: if all values positive and range > 1000x, use log to prevent squishing
    all_vals = np.concatenate([d for d in data if len(d) > 0])
    pos_vals = all_vals[all_vals > 0]
    if len(pos_vals) == len(all_vals) and len(pos_vals) > 0:
        val_range = pos_vals.max() / (pos_vals.min() + 1e-9)
        if val_range > 1000:
            ax.set_yscale("log")
            ax.set_ylabel("Value (log scale)", labelpad=12, fontsize=14, color=TEXT_COLOR)
        else:
            ax.set_ylabel("Value", labelpad=12, fontsize=14, color=TEXT_COLOR)
    else:
        ax.set_ylabel("Value", labelpad=12, fontsize=14, color=TEXT_COLOR)

    patches = [mpatches.Patch(color=col_colors[c], label=c) for c in cols]
    ax.legend(handles=patches, fontsize=11, loc="upper right",
              framealpha=0.3, facecolor=CARD_BG, edgecolor="#444",
              borderpad=0.8, labelspacing=0.5)


def _correlation_heatmap(ax, df, cols):
    corr_df, arr = _safe_corr(df, cols)
    np.fill_diagonal(arr, 0)
    corr_clean = pd.DataFrame(arr, index=corr_df.index, columns=corr_df.columns)
    orange_cmap = sns.light_palette("#FF6B00", as_cmap=True)
    n = len(cols)
    # Scale font sizes down for large matrices so annotations don't overlap
    annot_size  = max(5, 11 - n // 3)
    tick_size   = max(5, 10 - n // 4)
    annot = n <= 20   # skip annotations when >20 cols — too crowded
    sns.heatmap(corr_clean, ax=ax, annot=annot, fmt=".2f", cmap=orange_cmap,
                linewidths=0.4, linecolor="#0A0A0A",
                annot_kws={"size": annot_size, "color": "#111111", "weight": "bold"},
                vmin=0, vmax=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation Strength"})
    ax.set_title(f"Correlation Heatmap ({n} columns · darker = stronger)",
                 color=TEXT_COLOR, pad=14)
    ax.tick_params(axis="x", rotation=35, labelsize=tick_size, colors="#CCCCCC")
    ax.tick_params(axis="y", rotation=0,  labelsize=tick_size, colors="#CCCCCC")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)


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


def _bar_categorical(ax, series):
    vc = series.value_counts().head(15)
    bar_colors = [COLOR_POOL[i % len(COLOR_POOL)] for i in range(len(vc))]
    bars = ax.bar(vc.index.astype(str), vc.values,
                  color=bar_colors, alpha=0.88, edgecolor="#0A0A0A")
    ax.set_title(f"Top Categories – {series.name}", color=TEXT_COLOR, pad=12)
    ax.set_xlabel(series.name, labelpad=8)
    ax.set_ylabel("Count", labelpad=8)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
    ax.grid(True, axis="y")


def _pie(ax, series):
    vc = series.value_counts().head(7)
    shuffled = COLOR_POOL.copy()
    random.shuffle(shuffled)
    wedges, texts, autotexts = ax.pie(
        vc.values, labels=vc.index.astype(str),
        autopct="%1.1f%%", startangle=140,
        colors=shuffled[:len(vc)],
        wedgeprops=dict(edgecolor="#0A0A0A", linewidth=1.5),
        textprops=dict(color=TEXT_COLOR, fontsize=9))
    for at in autotexts:
        at.set_color("#111111")
        at.set_fontweight("bold")
    ax.set_title(f"Composition – {series.name}", color=TEXT_COLOR, pad=12)


# ─────────────── master generator ───────────────────────────────────────────

def make_chart(title, hint, path, col_map):
    """Helper to build a consistent chart dict."""
    return {"title": title, "hint": hint, "path": path, "colors": col_map}


def generate_charts(df):
    _base_style()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if "id" not in c.lower() and "index" not in c.lower()]
    cat_cols     = [c for c in df.select_dtypes(include=["object","category"]).columns
                    if df[c].nunique() <= 30]
    date_cols    = [c for c in df.columns
                    if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]

    col_colors = _assign_colors(list(df.columns))

    # chart_groups: list of { label, icon, charts: [make_chart(...),...] }
    chart_groups = []

    # ── 1. Box plot ──────────────────────────────────────────────────────────
    if numeric_cols:
        fig_w = max(16, len(numeric_cols) * 2.8)
        fig_h = 8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        _boxplot(ax, df, numeric_cols, col_colors)
        plt.tight_layout(pad=2.5)
        path = "static/chart_boxplot.png"
        os.makedirs("static", exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        chart_groups.append({
            "label": "Box Plot",
            "icon":  "📦",
            "scroll": "h",
            "charts": [make_chart(
                "Box Plot – All Numeric Columns",
                "Each colored box = one column. White line = median. Dots = outliers.",
                path, {c: col_colors[c] for c in numeric_cols}
            )]
        })

    # ── 2. Histograms (one per column) ───────────────────────────────────────
    if numeric_cols:
        hist_cards = []
        for gi, col in enumerate(numeric_cols):
            fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
            ax.set_facecolor(CARD_BG)
            _histogram(ax, df[col], col_colors[col])
            plt.tight_layout(pad=1.5)
            path = f"static/chart_hist_{gi}.png"
            _save(fig, path)
            hist_cards.append(make_chart(
                f"Histogram – {col}",
                "Distribution of values. White dashed = mean.",
                path, {col: col_colors[col]}
            ))
        chart_groups.append({
            "label": "Histograms",
            "icon":  "📊",
            "scroll": "v",
            "charts": hist_cards
        })

    # ── 3. Correlation heatmap — ALL numeric columns, dynamic sizing ────────
    if len(numeric_cols) >= 2:
        hmap_cols = numeric_cols          # no cap — show every column
        n = len(hmap_cols)
        sz = max(9, n * 0.9)             # grows with column count
        fig, ax = plt.subplots(figsize=(sz, sz * 0.88), facecolor=BG_COLOR)
        ax.set_facecolor(CARD_BG)
        _correlation_heatmap(ax, df, hmap_cols)
        plt.tight_layout(pad=1.5)
        path = "static/chart_correlation.png"
        _save(fig, path)
        chart_groups.append({
            "label": "Correlation Heatmap",
            "icon":  "🌡️",
            "scroll": "h",
            "charts": [make_chart(
                "Correlation Heatmap",
                "Darker orange = stronger correlation. All numeric columns shown.",
                path, {c: col_colors[c] for c in hmap_cols}
            )]
        })

    # ── 4. Scatter plots (all pairs, one per card) ───────────────────────────
    if len(numeric_cols) >= 2:
        scatter_cards = []
        for gi, (cx, cy) in enumerate(combinations(numeric_cols, 2)):
            fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
            ax.set_facecolor(CARD_BG)
            _scatter_single(ax, df, cx, cy, col_colors[cx], col_colors[cy])
            plt.tight_layout(pad=1.5)
            path = f"static/chart_scatter_{gi}.png"
            _save(fig, path)
            scatter_cards.append(make_chart(
                f"Scatter – {cx} × {cy}",
                "X-axis color = X column · Y-axis color = Y column.",
                path, {cx: col_colors[cx], cy: col_colors[cy]}
            ))
        chart_groups.append({
            "label": "Scatter Plots",
            "icon":  "🔵",
            "scroll": "v",
            "charts": scatter_cards
        })

    # ── 5. Categorical bars ──────────────────────────────────────────────────
    if cat_cols:
        cat_cards = []
        for gi, col in enumerate(cat_cols):
            fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
            ax.set_facecolor(CARD_BG)
            _bar_categorical(ax, df[col])
            plt.tight_layout(pad=1.5)
            path = f"static/chart_cat_{gi}.png"
            _save(fig, path)
            cat_cards.append(make_chart(
                f"Categories – {col}",
                "Bar height = frequency. Each bar has a distinct color.",
                path, {col: col_colors[col]}
            ))
        chart_groups.append({
            "label": "Category Counts",
            "icon":  "📋",
            "scroll": "v",
            "charts": cat_cards
        })

    # ── 6. Pie charts — skip date/time cols, clean legend, distinct colors ──
    PIE_COLORS = [
        "#FF6B6B","#FF9F43","#FECA57","#1DD1A1","#54A0FF",
        "#FF9FF3","#00D2D3","#2ECC71","#9B59B6","#E67E22",
    ]
    pie_eligible = [
        c for c in cat_cols
        if "date" not in c.lower()
        and "time" not in c.lower()
        and 2 <= df[c].nunique() <= 8
    ]
    if pie_eligible:
        pie_cards = []
        for gi, col in enumerate(pie_eligible[:5]):
            try:
                counts = df[col].value_counts().head(8)
                if len(counts) < 2:
                    continue
                pc = PIE_COLORS[:len(counts)]
                random.shuffle(pc)
                fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
                ax.set_facecolor(BG_COLOR)
                wedges, texts, autotexts = ax.pie(
                    counts.values, labels=None, autopct="%1.1f%%",
                    startangle=140, colors=pc,
                    wedgeprops=dict(edgecolor="#0A0A0A", linewidth=1.5),
                    pctdistance=0.78,
                )
                for at in autotexts:
                    at.set_color("#111111"); at.set_fontweight("bold"); at.set_fontsize(9)
                ax.legend(
                    wedges, counts.index.astype(str),
                    title=col, title_fontsize=9,
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, framealpha=0.15,
                    facecolor=CARD_BG, edgecolor="#333", labelcolor=TEXT_COLOR,
                )
                ax.set_title(f"{col} – Composition", color=TEXT_COLOR, pad=14, fontsize=12)
                plt.tight_layout(pad=1.5)
                safe = "".join(x if x.isalnum() else "_" for x in col)
                path = f"static/chart_pie_{safe}_{gi}.png"
                _save(fig, path)
                pie_cards.append(make_chart(
                    f"Composition – {col}",
                    "Each slice = one category. Legend shows labels. Percentage inside.",
                    path, {col: col_colors[col]}
                ))
            except Exception:
                continue
        if pie_cards:
            chart_groups.append({
                "label": "Pie Charts",
                "icon":  "🥧",
                "scroll": "v",
                "charts": pie_cards
            })

    # ── 7. Time-series ───────────────────────────────────────────────────────
    if date_cols and numeric_cols:
        time_col  = date_cols[0]
        value_col = numeric_cols[0]
        try:
            tmp = df[[time_col, value_col]].copy()
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
            tmp = tmp.dropna().sort_values(time_col).head(500)
            if len(tmp) > 5:
                lc = col_colors[value_col]
                fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_COLOR)
                ax.set_facecolor(CARD_BG)
                ax.plot(tmp[time_col], tmp[value_col], color=lc, lw=2.2, alpha=0.9)
                ax.fill_between(tmp[time_col], tmp[value_col], alpha=0.18, color=lc)
                ax.set_title(f"Trend Over Time – {value_col}", color=TEXT_COLOR, pad=12)
                ax.set_xlabel(time_col, labelpad=8)
                ax.set_ylabel(value_col, color=lc, labelpad=8)
                ax.tick_params(axis="y", colors=lc)
                ax.tick_params(axis="x", rotation=30)
                ax.grid(True)
                plt.tight_layout(pad=1.5)
                path = "static/chart_trend.png"
                _save(fig, path)
                chart_groups.append({
                    "label": "Time-Series",
                    "icon":  "📈",
                    "scroll": "h",
                    "charts": [make_chart(
                        f"Trend – {value_col}",
                        f"How {value_col} changes over time.",
                        path, {value_col: lc}
                    )]
                })
        except Exception:
            pass

    return chart_groups, col_colors


# ─────────────── routes ──────────────────────────────────────────────────────

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
        df    = next(chunk)
        df    = df.drop_duplicates()
        df    = df.fillna("N/A")
    except Exception as e:
        return f"Error reading file: {e}"

    df_num = df.replace("N/A", np.nan)

    chart_groups, col_colors = generate_charts(df_num)
    table  = df.head(10).to_html(classes="table table-bordered", index=False)
    stats  = df_num.describe(include="all").round(2).to_html(
                 classes="table table-bordered")

    summary = {
        "rows":            df.shape[0],
        "columns":         df.shape[1],
        "column_names":    list(df.columns),
        "numeric_columns": df_num.select_dtypes(include="number").columns.tolist(),
        "missing_values":  df.isnull().sum().to_dict(),
    }

    return render_template('result.html',
                           table=table,
                           stats=stats,
                           chart_groups=chart_groups,
                           summary=summary,
                           col_colors=col_colors)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
