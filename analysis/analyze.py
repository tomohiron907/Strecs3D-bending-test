#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三点曲げ試験データ解析: 均一インフィル(normal) vs Strecs3D最適化インフィル(strecs)

入力 : ../data/*.csv  (Shift_JIS, 列 = 試験回数, 時間[sec], 移動量[mm], 荷重[N])
出力 : figures/*.png|pdf (修論用/日本語), figures_journal/*.pdf|png|tiff (投稿用/英語),
       processed/*.csv,  summary.csv, summary_group.csv, summary.md

試験片・治具諸元 (修士論文 4.1.2 より)
  外形       : 180 x 50 x 7.5 mm (平置きで造形)
  試験の向き : 縦置き (edgewise)。荷重は 50 mm 方向 (造形面内) に加える。
               FEA の荷重方向と同じ。
  支点間距離 : L = 110 mm
  幅         : b = 7.5 mm  (荷重方向に直交)
  高さ       : h = 50 mm   (荷重方向)

三点曲げの公称式 (ASTM D790 / JIS K 7171 準拠)
  曲げ応力   sigma = 3 F L / (2 b h^2)
  曲げひずみ eps   = 6 D h / L^2
  曲げ弾性率 E     = L^3 m / (4 b h^3)      m = 荷重-たわみ曲線の初期勾配 [N/mm]

※ 供試体は内部がインフィル(空隙を含む)であるため、上式で得られる値は
   外形寸法を基準とした「見かけの(等価)曲げ特性」である。外形・質量が
   共通な 2 条件の相対比較を目的とするため、この定義で一貫して扱う。
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

# ----------------------------------------------------------------------------
# 0. 設定
# ----------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
FIG_DIR  = os.path.join(HERE, "figures")           # 修論用 (日本語)
FIG_DIR_EN = os.path.join(HERE, "figures_journal")  # 投稿用 (英語・ジャーナル体裁)
PROC_DIR = os.path.join(HERE, "processed")
for d in (FIG_DIR, FIG_DIR_EN, PROC_DIR):
    os.makedirs(d, exist_ok=True)

SPAN_L = 110.0   # mm  支点間距離
WIDTH_B = 7.5    # mm  幅 (荷重方向に直交; 縦置き試験なので板厚)
HEIGHT_H = 50.0  # mm  高さ (荷重方向; 縦置き試験なので板幅)

ENCODINGS = ("shift_jis", "cp932", "utf-8-sig", "utf-8")

# トウ補正・初期勾配同定のパラメータ
WIN_MM      = 0.40    # 勾配探索の窓幅 [mm]
F_LO        = 0.10    # 窓の開始荷重の下限 (ピーク荷重比) — 接触遷移を除外
F_HI        = 0.75    # 窓の終了荷重の上限 (ピーク荷重比)
R2_MIN      = 0.9995  # 直線性のしきい値
PRE_CONTACT_MM = 6.0  # この変位未満は確実に非接触 → ゼロ点補正(タレ)に使用

# 中断された(=ピークに達していない)試験。集計平均・作図の両方から除外する。
ABORTED = set()

# ----------------------------------------------------------------------------
# 1. 日本語フォント
# ----------------------------------------------------------------------------
def setup_font():
    cands = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                return name
            except Exception:
                continue
    return None

FONT = setup_font()
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.dpi": 130, "savefig.dpi": 300,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.9, "savefig.bbox": "tight",
})

C_NORMAL = "#4A5A9E"   # 均一インフィル
C_STRECS = "#C1553B"   # Strecs3D

# ----------------------------------------------------------------------------
# 2. 読み込み
# ----------------------------------------------------------------------------
def read_csv_any(path):
    """日本語計測器出力(Shift_JIS想定)を文字コード自動判定で読む。"""
    last = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            if any("�" in str(c) for c in df.columns):
                raise UnicodeDecodeError(enc, b"", 0, 1, "replacement char")
            return df, enc
        except (UnicodeDecodeError, LookupError) as e:
            last = e
    raise RuntimeError(f"{path}: 文字コード判定に失敗 ({last})")


def load_specimen(path):
    df, enc = read_csv_any(path)
    # 列名を位置で正規化: 試験回数, 時間, 移動量, 荷重
    df = df.iloc[:, :4].copy()
    df.columns = ["cycle", "time_s", "disp_raw_mm", "load_raw_N"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["disp_raw_mm", "load_raw_N"])
    df = df.sort_values("disp_raw_mm", kind="stable").reset_index(drop=True)

    name = os.path.splitext(os.path.basename(path))[0]
    group = "strecs" if name.startswith("strecs") else "normal"
    return dict(name=name, group=group, encoding=enc, df=df)


# ----------------------------------------------------------------------------
# 3. 前処理: ゼロ点補正 + トウ補正(初期の遊び/接触立ち上がりの除去)
# ----------------------------------------------------------------------------
def preprocess(sp):
    """
    (a) タレ補正 : 非接触区間の平均荷重を差し引き、荷重ゼロ点を較正する。
    (b) トウ補正 : 圧子が試験片に接触するまでの空走(荷重ゼロで変位のみ進む区間)と、
                   接触初期の非線形な当たり(位置決め窪みへの着座)を除去する。
                   ASTM D790 の toe compensation に準拠し、荷重-変位曲線の
                   最急直線部を求め、その接線を荷重ゼロへ外挿した点を
                   真のたわみ原点 D0 とする。
    """
    d = sp["df"]["disp_raw_mm"].to_numpy(float)
    F = sp["df"]["load_raw_N"].to_numpy(float)

    # --- (a) タレ補正 ---
    pre = F[d < PRE_CONTACT_MM]
    tare = float(pre.mean()) if pre.size >= 10 else 0.0
    F = F - tare

    Fpeak_raw = float(F.max())

    # --- (b) 最急直線部の探索 ---
    step = float(np.median(np.diff(d)))
    w = max(6, int(round(WIN_MM / step)))

    best = None   # (slope, intercept, r2, i0, i1)
    for i in range(len(d) - w):
        j = i + w
        if F[i] < F_LO * Fpeak_raw or F[j] > F_HI * Fpeak_raw:
            continue
        x, y = d[i:j + 1], F[i:j + 1]
        m, b = np.polyfit(x, y, 1)
        if m <= 0:
            continue
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        if r2 < R2_MIN:
            continue
        if best is None or m > best[0]:
            best = (float(m), float(b), r2, i, j)

    if best is None:
        raise RuntimeError(f"{sp['name']}: 直線区間を同定できませんでした")

    slope, intercept, r2, i0, i1 = best
    D0 = -intercept / slope            # 荷重ゼロへの外挿点 = 補正後の原点

    disp = d - D0                      # 補正後たわみ [mm]

    # 補正後に有効な区間(たわみ>=0)のみを解析対象にする
    keep = disp >= 0.0
    # 立ち上がり前の負側ノイズは落とすが、原点は (0,0) を明示的に付与する
    disp = np.concatenate(([0.0], disp[keep]))
    F_c  = np.concatenate(([0.0], F[keep]))
    time = np.concatenate(([np.nan], sp["df"]["time_s"].to_numpy(float)[keep]))

    sp.update(
        tare_N=tare, D0_mm=float(D0), slope_N_per_mm=slope, fit_r2=r2,
        fit_range_mm=(float(d[i0] - D0), float(d[i1] - D0)),
        fit_range_Fratio=(float(F[i0] / Fpeak_raw), float(F[i1] / Fpeak_raw)),
        disp=disp, load=F_c, time=time,
    )
    return sp


# ----------------------------------------------------------------------------
# 4. 応力・ひずみ換算と特性値
# ----------------------------------------------------------------------------
def to_stress_strain(sp):
    L, b, h = SPAN_L, WIDTH_B, HEIGHT_H
    sp["stress_MPa"] = 3.0 * sp["load"] * L / (2.0 * b * h ** 2)     # N/mm^2 = MPa
    sp["strain"]     = 6.0 * sp["disp"] * h / (L ** 2)               # 無次元
    return sp


def metrics(sp):
    s, e = sp["stress_MPa"], sp["strain"]
    F, D = sp["load"], sp["disp"]

    ipk = int(np.argmax(s))
    # 曲げ弾性率 E = L^3 m / (4 b h^3)
    E = SPAN_L ** 3 * sp["slope_N_per_mm"] / (4.0 * WIDTH_B * HEIGHT_H ** 3)

    # 破壊/最大荷重までの吸収エネルギー(仕事) [J] と 靱性(単位体積あたり) [MJ/m^3]
    work_J   = float(np.trapezoid(F[:ipk + 1], D[:ipk + 1])) / 1000.0   # N*mm -> J
    tough    = float(np.trapezoid(s[:ipk + 1], e[:ipk + 1]))            # MPa = MJ/m^3

    return dict(
        name=sp["name"], group=sp["group"], encoding=sp["encoding"],
        tare_N=sp["tare_N"], D0_mm=sp["D0_mm"], fit_r2=sp["fit_r2"],
        stiffness_N_per_mm=sp["slope_N_per_mm"],
        E_flex_MPa=E,
        sigma_max_MPa=float(s[ipk]),
        strain_at_max=float(e[ipk]),
        F_max_N=float(F[ipk]),
        D_at_Fmax_mm=float(D[ipk]),
        work_to_peak_J=work_J,
        toughness_MJ_per_m3=tough,
        aborted=sp["name"] in ABORTED,
    )


# ----------------------------------------------------------------------------
# 5. 実行
# ----------------------------------------------------------------------------
def natural_key(p):
    n = os.path.basename(p)
    return (0 if n.startswith("normal") else 1, n)

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=natural_key)
specs = []
for p in paths:
    sp = load_specimen(p)
    preprocess(sp)
    to_stress_strain(sp)
    specs.append(sp)

    out = pd.DataFrame({
        "time_s": sp["time"],
        "disp_raw_mm": np.nan,
        "deflection_mm": sp["disp"],
        "load_N": sp["load"],
        "strain": sp["strain"],
        "stress_MPa": sp["stress_MPa"],
    })
    out.to_csv(os.path.join(PROC_DIR, f"{sp['name']}_processed.csv"),
               index=False, encoding="utf-8-sig", float_format="%.6g")

# 中断試験は作図対象から除外する (集計と同様、図にも載せない)
PLOT_SPECS = [sp for sp in specs if sp["name"] not in ABORTED]

res = pd.DataFrame([metrics(sp) for sp in specs])
res.to_csv(os.path.join(HERE, "summary.csv"), index=False, encoding="utf-8-sig",
           float_format="%.6g")

# ---- 群統計 (中断試験は除外) ----
valid = res[~res.aborted]
agg = valid.groupby("group").agg(
    n=("name", "count"),
    E_mean=("E_flex_MPa", "mean"), E_sd=("E_flex_MPa", "std"),
    k_mean=("stiffness_N_per_mm", "mean"), k_sd=("stiffness_N_per_mm", "std"),
    s_mean=("sigma_max_MPa", "mean"), s_sd=("sigma_max_MPa", "std"),
    F_mean=("F_max_N", "mean"), F_sd=("F_max_N", "std"),
    e_mean=("strain_at_max", "mean"), e_sd=("strain_at_max", "std"),
    W_mean=("work_to_peak_J", "mean"), W_sd=("work_to_peak_J", "std"),
    T_mean=("toughness_MJ_per_m3", "mean"), T_sd=("toughness_MJ_per_m3", "std"),
).reindex(["normal", "strecs"])
agg.to_csv(os.path.join(HERE, "summary_group.csv"), encoding="utf-8-sig",
           float_format="%.6g")

# ----------------------------------------------------------------------------
# 6. 作図  (図はスタイル2種を生成: 修論用[日本語] / 投稿用[英語・ジャーナル体裁])
# ----------------------------------------------------------------------------
LBL = {"normal": "均一インフィル (14 %)", "strecs": "Strecs3D 最適化"}  # summary.md 用


def register_font(paths):
    """フォントを matplotlib に登録し、family 名を返す。"""
    name = None
    for p in paths:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                if name is None:
                    name = font_manager.FontProperties(fname=p).get_name()
            except Exception:
                continue
    return name


# 投稿規定で最も広く受理される Arial / Helvetica を優先
ARIAL = register_font([
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
]) or "DejaVu Sans"

# 修論用 (これまでの体裁)
RC_JA = {
    "font.family": FONT or "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 130, "savefig.dpi": 300,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# 投稿用 (Elsevier / Springer 系の一般的な図版規定に合わせる)
#   ・本文フォントと揃う sans-serif (Arial) / 実寸 7-8 pt
#   ・単段 85 mm・全段 180 mm の実寸で作成し、縮小せずそのまま貼れる
#   ・枠付き・目盛内向き・補助目盛あり・グリッドなし
#   ・Type 3 フォント禁止規定に対応 (pdf/ps.fonttype = 42)
RC_EN = {
    "font.family": "sans-serif",
    "font.sans-serif": [ARIAL, "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": ARIAL, "mathtext.it": f"{ARIAL}:italic", "mathtext.bf": f"{ARIAL}:bold",
    "axes.unicode_minus": False,
    "figure.dpi": 130, "savefig.dpi": 600,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.linewidth": 0.7, "lines.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "legend.frameon": False, "legend.handlelength": 2.0,
    "legend.labelspacing": 0.30, "legend.borderpad": 0.2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# mm → inch (単段 85 mm / 全段 180 mm)
MM = 1.0 / 25.4
W1, W2 = 85 * MM, 180 * MM

TXT_JA = dict(
    raw_x="移動量(生データ) [mm]", load_y="荷重 [N]", load_yF="荷重 $F$ [N]",
    defl_x="たわみ $D$ [mm]",
    strain_x="曲げひずみ $\\varepsilon$ [%]", stress_y="曲げ応力 $\\sigma$ [MPa]",
    f1a_title="(a) 生データ — 圧子空走区間を含む",
    f1a_note="荷重が立たない空走区間\n(圧子が試験片に未接触)",
    f1b_title="(b) トウ補正 — 最急直線部を荷重ゼロへ外挿",
    f1b_curve="{name} (タレ補正後)", f1b_fit="最急直線部 (r²={r2:.5f})",
    f1b_d0="補正原点 $D_0$ = {d0:.3f} mm",
    f1_sup="図1  前処理: ゼロ点(タレ)補正とトウ補正",
    f2_title="図2  荷重–たわみ曲線 (トウ補正後)",
    f3_title="図3  曲げ応力–ひずみ曲線 (見かけ)", f3_legtitle="試験片", f3_note=True,
    f4_title="図4  群平均の曲げ応力–ひずみ曲線 (帯は実測範囲)",
    f4_callout="最大曲げ応力  ×{rs:.2f}\n見かけ曲げ弾性率  ×{rE:.2f}",
    f5_sup="図5  機械的特性の比較 (平均 ± 標準偏差, 点は各試験片)",
    f5_xticks=["均一\n14 %", "Strecs3D"],
    f5_items=[("見かけ曲げ弾性率 $E$", "MPa"), ("最大曲げ応力 $\\sigma_{max}$", "MPa"),
              ("曲げ剛性 $F/D$", "N/mm"), ("靱性 (ピークまで)", "MJ/m³")],
    grp={"normal": "均一インフィル (14 %)", "strecs": "Strecs3D 最適化"},
    grp_n="{g}  (n={n})",
)

TXT_EN = dict(
    raw_x="Crosshead displacement (mm)", load_y="Load (N)", load_yF="Load, $F$ (N)",
    defl_x="Deflection, $D$ (mm)",
    strain_x="Flexural strain, $\\varepsilon$ (%)",
    stress_y="Flexural stress, $\\sigma$ (MPa)",
    f1a_title="", f1a_note="Take-up region\n(indenter not yet in contact)",
    f1b_title="",
    f1b_curve="{name} (tare-corrected)", f1b_fit="Steepest linear fit ($r^2$ = {r2:.5f})",
    f1b_d0="Corrected origin\n$D_0$ = {d0:.3f} mm",
    f1_sup="",
    f2_title="", f3_title="", f3_legtitle=None, f3_note=False, f4_title="",
    f4_callout="$\\sigma_{{max}}$  ×{rs:.2f}\n$E$  ×{rE:.2f}",
    f5_sup="", f5_xticks=["Uniform\n(14%)", "Strecs3D"],
    f5_items=[("Apparent modulus", "$E$ (MPa)"),
              ("Flexural strength", "$\\sigma_{max}$ (MPa)"),
              ("Bending stiffness", "$F/D$ (N mm$^{-1}$)"),
              ("Toughness", "(MJ m$^{-3}$)")],
    grp={"normal": "Uniform infill (14%)", "strecs": "Strecs3D-optimized"},
    grp_n="{g}, $n$ = {n}",
)

# 英語版の試験片ID (normal1 → Uniform-1 など)
def en_name(sp):
    same = [s for s in PLOT_SPECS if s["group"] == sp["group"]]
    k = same.index(sp) + 1
    return ("Uniform" if sp["group"] == "normal" else "Strecs3D") + f"-{k}"

STYLES = {
    "ja": dict(dir=FIG_DIR, rc=RC_JA, txt=TXT_JA, fmts=("png", "pdf"),
               cn="#4A5A9E", cs="#C1553B",
               name=lambda sp: sp["name"], panel=False, group_legend=False,
               sizes=dict(f1=(11.2, 4.3), f2=(6.6, 4.8), f3=(7.0, 5.2),
                          f4=(7.0, 5.2), f5=(13.6, 3.9))),
    # Okabe-Ito のカラーユニバーサル配色。グレースケール印刷でも線種で判別可
    "en": dict(dir=FIG_DIR_EN, rc=RC_EN, txt=TXT_EN,
               fmts=("pdf", "png", "tiff"),
               cn="#0072B2", cs="#D55E00",
               name=en_name, panel=True, group_legend=True,
               sizes=dict(f1=(W2, 2.75), f2=(W1, 2.55), f3=(W1, 2.75),
                          f4=(W1, 2.75), f5=(W2, 2.05))),
}


def make_figures(cfg):
    txt, out, C_N, C_S = cfg["txt"], cfg["dir"], cfg["cn"], cfg["cs"]
    os.makedirs(out, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update(cfg["rc"])
    lw = plt.rcParams["lines.linewidth"]

    def color_of(sp):
        return C_S if sp["group"] == "strecs" else C_N

    def style_of(sp):
        same = [s for s in PLOT_SPECS if s["group"] == sp["group"]]
        k = same.index(sp)
        return dict(color=color_of(sp), ls=["-", "--", "-."][k % 3], lw=lw)

    def legend_of(ax, **kw):
        """群凡例 (投稿用) / 試験片凡例 (修論用)。"""
        if not cfg["group_legend"]:
            return ax.legend(**kw)
        h = []
        for g in ("normal", "strecs"):
            n = sum(1 for s in PLOT_SPECS if s["group"] == g)
            h.append(Line2D([], [], color=C_N if g == "normal" else C_S, lw=lw,
                            label=txt["grp_n"].format(g=txt["grp"][g], n=n)))
        return ax.legend(handles=h, **kw)

    def panel(ax, letter, dx=-0.16, dy=1.04):
        if cfg["panel"]:
            ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="bottom", ha="left")

    def title(ax, s):
        if s:
            ax.set_title(s)

    def save(fig, stem):
        """投稿用は仕上がり幅を段組幅 (85 / 180 mm) ちょうどに合わせて保存する。

        savefig(bbox_inches="tight") は余白を切り詰めるため、figsize をそのまま
        指定しても出力幅は段組幅より小さくなる。描画→タイトな外接矩形を実測→
        不足分だけ figsize を広げる、を数回繰り返して収束させ、最後に矩形を
        左右対称に伸ばして目標幅ちょうどの bbox で書き出す。
        """
        kwargs = {}
        if cfg["panel"]:
            target = cfg["sizes"]["f" + stem[3]][0]           # inch (段組幅)
            pad = plt.rcParams["savefig.pad_inches"]
            for _ in range(6):
                fig.canvas.draw()
                bb = fig.get_tightbbox(fig.canvas.get_renderer())
                gap = target - (bb.width + 2 * pad)
                if abs(gap) < 0.004:                          # 0.1 mm
                    break
                w, h = fig.get_size_inches()
                fig.set_size_inches(w + gap, h)
            fig.canvas.draw()
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            ext_w = (target - 2 * pad - bb.width) / 2.0
            kwargs = dict(pad_inches=0, bbox_inches=Bbox.from_extents(
                bb.x0 - pad - ext_w, bb.y0 - pad, bb.x1 + pad + ext_w, bb.y1 + pad))
        for ext in cfg["fmts"]:
            kw = dict(pil_kwargs={"compression": "tiff_lzw"}) if ext == "tiff" else {}
            fig.savefig(os.path.join(out, f"{stem}.{ext}"), **kw, **kwargs)
        plt.close(fig)

    # --- Fig.1 前処理の妥当性 ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=cfg["sizes"]["f1"])
    ax = axes[0]
    for sp in PLOT_SPECS:
        ax.plot(sp["df"]["disp_raw_mm"], sp["df"]["load_raw_N"],
                **style_of(sp), label=cfg["name"](sp))
    ax.set_xlabel(txt["raw_x"]); ax.set_ylabel(txt["load_y"])
    title(ax, txt["f1a_title"]); panel(ax, "a")
    legend_of(ax, frameon=False, loc="upper left")
    ax.axvspan(0, min(s["D0_mm"] for s in PLOT_SPECS), color="0.88", alpha=0.6,
               zorder=0)
    ax.annotate(txt["f1a_note"], xy=(3.1, 0.10 * res.F_max_N.max()), ha="center",
                fontsize=plt.rcParams["legend.fontsize"], color="0.35")

    ax = axes[1]
    sp = PLOT_SPECS[0]
    d = sp["df"]["disp_raw_mm"].to_numpy()
    F = sp["df"]["load_raw_N"].to_numpy() - sp["tare_N"]
    m = (d > sp["D0_mm"] - 0.6) & (d < sp["D0_mm"] + 2.2)
    ax.plot(d[m], F[m], color=C_N, lw=lw,
            label=txt["f1b_curve"].format(name=cfg["name"](sp)))
    xs = np.array([sp["D0_mm"] - 0.3, sp["D0_mm"] + 2.2])
    ax.plot(xs, sp["slope_N_per_mm"] * (xs - sp["D0_mm"]), color="k", ls="--",
            lw=lw * 0.7, label=txt["f1b_fit"].format(r2=sp["fit_r2"]))
    ax.axvline(sp["D0_mm"], color=C_S, lw=lw * 0.65)
    ax.plot([sp["D0_mm"]], [0], "o", color=C_S, ms=4.5, zorder=5)
    ax.annotate(txt["f1b_d0"].format(d0=sp["D0_mm"]), xy=(sp["D0_mm"], 0),
                xytext=(sp["D0_mm"] + 0.30, 22), color=C_S,
                fontsize=plt.rcParams["legend.fontsize"],
                arrowprops=dict(arrowstyle="->", color=C_S, lw=0.8))
    ax.set_xlabel(txt["raw_x"]); ax.set_ylabel(txt["load_y"])
    title(ax, txt["f1b_title"]); panel(ax, "b")
    ax.legend(frameon=False, loc="upper left")
    if txt["f1_sup"]:
        fig.suptitle(txt["f1_sup"], y=1.02, fontsize=13)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig1_preprocessing")

    # --- Fig.2 荷重-たわみ曲線 ----------------------------------------------
    fig, ax = plt.subplots(figsize=cfg["sizes"]["f2"])
    for sp in PLOT_SPECS:
        ax.plot(sp["disp"], sp["load"], **style_of(sp), label=cfg["name"](sp))
    for g, c in (("normal", C_N), ("strecs", C_S)):
        sub = valid[valid.group == g]
        ax.plot(sub.D_at_Fmax_mm, sub.F_max_N, "o", color=c, ms=4.5, mec="w",
                mew=0.8, zorder=6)
    ax.set_xlabel(txt["defl_x"]); ax.set_ylabel(txt["load_yF"])
    title(ax, txt["f2_title"])
    legend_of(ax, frameon=False, loc="upper left")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    save(fig, "fig2_load_deflection")

    # --- Fig.3 応力-ひずみ曲線 (主図) ---------------------------------------
    fig, ax = plt.subplots(figsize=cfg["sizes"]["f3"])
    for sp in PLOT_SPECS:
        ax.plot(sp["strain"] * 100, sp["stress_MPa"], **style_of(sp),
                label=cfg["name"](sp))
    for _, r in valid.iterrows():
        c = C_S if r.group == "strecs" else C_N
        ax.plot(r.strain_at_max * 100, r.sigma_max_MPa, "o", color=c, ms=4.5,
                mec="w", mew=0.8, zorder=6)
    ax.set_xlabel(txt["strain_x"]); ax.set_ylabel(txt["stress_y"])
    title(ax, txt["f3_title"])
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    leg = legend_of(ax, frameon=False, loc="lower right", title=txt["f3_legtitle"])
    leg._legend_box.align = "left"
    ax.add_artist(leg)
    if txt.get("f3_note", True):
        ax.text(0.03, 0.97,
                f"$L$ = {SPAN_L:.0f} mm,  $b$ = {WIDTH_B:.1f} mm,  $h$ = {HEIGHT_H:.0f} mm\n"
                f"$\\sigma = 3FL/2bh^2$,   $\\varepsilon = 6Dh/L^2$",
                transform=ax.transAxes, va="top", color="0.35",
                fontsize=plt.rcParams["legend.fontsize"])
    save(fig, "fig3_stress_strain")

    # --- Fig.4 群平均 ± 実測範囲 ---------------------------------------------
    fig, ax = plt.subplots(figsize=cfg["sizes"]["f4"])
    grid = np.linspace(0, min(sp["strain"].max() for sp in PLOT_SPECS), 600)
    for g, c in (("normal", C_N), ("strecs", C_S)):
        curves = [np.interp(grid, sp["strain"], sp["stress_MPa"])
                  for sp in PLOT_SPECS if sp["group"] == g]
        A = np.vstack(curves)
        ax.fill_between(grid * 100, A.min(0), A.max(0), color=c, alpha=0.20, lw=0)
        ax.plot(grid * 100, A.mean(0), color=c, lw=lw * 1.25,
                label=txt["grp_n"].format(g=txt["grp"][g], n=len(curves)))
    ax.set_xlabel(txt["strain_x"]); ax.set_ylabel(txt["stress_y"])
    title(ax, txt["f4_title"])
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    r_s = agg.loc["strecs", "s_mean"] / agg.loc["normal", "s_mean"]
    r_E = agg.loc["strecs", "E_mean"] / agg.loc["normal", "E_mean"]
    ax.text(0.03, 0.97, txt["f4_callout"].format(rs=r_s, rE=r_E),
            transform=ax.transAxes, va="top",
            fontsize=plt.rcParams["legend.fontsize"] * 1.05,
            bbox=None if cfg["panel"] else
                 dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.6))
    save(fig, "fig4_mean_stress_strain")

    # --- Fig.5 特性値の比較 --------------------------------------------------
    cols = ["E_flex_MPa", "sigma_max_MPa", "stiffness_N_per_mm", "toughness_MJ_per_m3"]
    fig, axes = plt.subplots(1, 4, figsize=cfg["sizes"]["f5"])
    fig.subplots_adjust(wspace=0.42 if not cfg["panel"] else 0.62)
    for k, (ax, col, (ttl, unit)) in enumerate(zip(axes, cols, txt["f5_items"])):
        means, sds, pts = [], [], []
        for g in ("normal", "strecs"):
            v = valid.loc[valid.group == g, col].to_numpy()
            means.append(v.mean()); sds.append(v.std(ddof=1) if v.size > 1 else 0.0)
            pts.append(v)
        x = [0, 1]
        ax.bar(x, means, width=0.58, color=[C_N, C_S], alpha=0.85, yerr=sds,
               capsize=3 if cfg["panel"] else 5,
               error_kw=dict(lw=0.9 if cfg["panel"] else 1.2, ecolor="0.25"))
        for xi, v in zip(x, pts):
            ax.plot(np.full(v.size, xi), v, "o", color="0.15",
                    ms=3.2 if cfg["panel"] else 4.5, zorder=5)
        ax.set_xticks(x); ax.set_xticklabels(txt["f5_xticks"])
        ax.set_title(ttl); ax.set_ylabel(unit, labelpad=2)
        ax.set_ylim(0, max(means) * 1.38)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        panel(ax, "abcd"[k], dx=-0.40, dy=1.02)
        ratio = means[1] / means[0]
        ax.annotate(f"×{ratio:.2f}", xy=(0.5, max(means) * 1.22), ha="center",
                    fontsize=9 if cfg["panel"] else 13, fontweight="bold", color=C_S)
        ax.annotate("", xy=(1, max(means) * 1.14), xytext=(0, max(means) * 1.14),
                    arrowprops=dict(arrowstyle="->", color=C_S,
                                    lw=0.9 if cfg["panel"] else 1.4))
    if txt["f5_sup"]:
        fig.suptitle(txt["f5_sup"], y=1.05, fontsize=13)
    save(fig, "fig5_metrics")


for _cfg in STYLES.values():
    make_figures(_cfg)

# ----------------------------------------------------------------------------
# 7. Markdown サマリ
# ----------------------------------------------------------------------------
def fmt(v, p=3):
    return "—" if pd.isna(v) else f"{v:.{p}f}"

lines = []
lines.append("# 三点曲げ試験 解析結果 — 均一インフィル vs Strecs3D 最適化\n")
lines.append(f"- 支点間距離 L = {SPAN_L:.0f} mm / 幅 b = {WIDTH_B:.1f} mm / 高さ h = {HEIGHT_H:.0f} mm (縦置き試験)")
lines.append(f"- 文字コード: {', '.join(sorted(set(res.encoding)))} で読み込み")
lines.append("- 曲げ応力 σ = 3FL/(2bh²)、曲げひずみ ε = 6Dh/L²、曲げ弾性率 E = L³m/(4bh³)")
lines.append("- 空走区間はタレ補正 + トウ補正(ASTM D790)で除去\n")

lines.append("## 試験片ごとの結果\n")
lines.append("| 試験片 | 群 | 補正原点 D₀ [mm] | 剛性 [N/mm] | E [MPa] | F_max [N] | σ_max [MPa] | ε@σmax [%] | 靱性 [MJ/m³] |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for _, r in res.iterrows():
    tag = " ⚠️中断" if r.aborted else ""
    lines.append(f"| {r['name']}{tag} | {LBL[r.group]} | {fmt(r.D0_mm)} | {fmt(r.stiffness_N_per_mm,1)} | "
                 f"{fmt(r.E_flex_MPa,0)} | {fmt(r.F_max_N,1)} | {fmt(r.sigma_max_MPa,2)} | "
                 f"{fmt(r.strain_at_max*100,2)} | {fmt(r.toughness_MJ_per_m3,3)} |")

lines.append("\n## 群平均 (中断試験を除く)\n")
lines.append("| 指標 | 均一インフィル 14 % | Strecs3D | 比 |")
lines.append("|---|---|---|---|")
rows = [("見かけ曲げ弾性率 E [MPa]", "E_mean", "E_sd", 0),
        ("曲げ剛性 F/D [N/mm]",     "k_mean", "k_sd", 1),
        ("最大荷重 F_max [N]",       "F_mean", "F_sd", 1),
        ("最大曲げ応力 σ_max [MPa]", "s_mean", "s_sd", 2),
        ("σ_max 時ひずみ [%]",       "e_mean", "e_sd", 3),
        ("ピークまでの仕事 [J]",     "W_mean", "W_sd", 3),
        ("靱性 [MJ/m³]",             "T_mean", "T_sd", 3)]
for label, mk, sk, p in rows:
    n_, s_ = agg.loc["normal", mk], agg.loc["strecs", mk]
    nsd, ssd = agg.loc["normal", sk], agg.loc["strecs", sk]
    scale = 100 if "ひずみ" in label else 1
    lines.append(f"| {label} | {fmt(n_*scale,p)} ± {fmt(nsd*scale,p)} | "
                 f"{fmt(s_*scale,p)} ± {fmt(ssd*scale,p)} | **×{s_/n_:.2f}** |")

open(os.path.join(HERE, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

print(f"[font] {FONT}")
print(res.to_string(index=False))
print()
print(agg.to_string())
print(f"\n出力: {FIG_DIR}, {PROC_DIR}, summary.csv, summary_group.csv, summary.md")
