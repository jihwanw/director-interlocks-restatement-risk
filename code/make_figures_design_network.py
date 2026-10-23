#!/usr/bin/env python3
"""
make_figures_design_network.py — two reader-facing figures for the Director Interlocks paper (added 2026-10-23).

  figures_k/figK6_design.pdf   : the dated, directed interlock and the paper's windows and exposure classes, drawn on the
                                 verified Toro -> Ecolab appointment (Jeffrey Ettinger, 4 May 2017). Every date is read from
                                 the Audit Analytics-derived files below; nothing is typed in.
  figures_k/figK7_network.pdf  : (a) the primary-sample interlock network alive on 1 January 2017 (largest component, 400
                                 highest-degree firms) with firms that filed a restatement in the previous five years marked;
                                 (b) the two-step neighbourhood of Ecolab at the 4 May 2017 appointment; (c) the same for
                                 K12 Inc. at the 13 April 2017 appointment from MUFG Americas Holdings.

Inputs (primary sample, suffix _art): data/edge_lifecycles_k_art.csv, data/tenure_spells_k_art.csv,
        data/restatements_with_dates.csv, data/pair_base_k_art.csv.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

HERE = Path(__file__).resolve().parent
D = HERE.parent / "data"
SRC = HERE.parent.parent / "3rd_revision" / "source"
FIG = SRC / "figures_k"; FIG.mkdir(parents=True, exist_ok=True)
RES = HERE.parent / "results_rebuild"; RES.mkdir(exist_ok=True)

def norm(s): return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")

edges = pd.read_csv(D / "edge_lifecycles_k_art.csv", dtype={"company_a": str, "company_b": str, "origin": str})
for c in ("company_a", "company_b", "origin"): edges[c] = norm(edges[c])
edges["edge_birth"] = pd.to_datetime(edges["edge_birth"]); edges["edge_death"] = pd.to_datetime(edges["edge_death"])
spells = pd.read_csv(D / "tenure_spells_k_art.csv", dtype={"company": str}); spells["company"] = norm(spells["company"])
spells["start"] = pd.to_datetime(spells["start"]); spells["end"] = pd.to_datetime(spells["end"])
rest = pd.read_csv(D / "restatements_with_dates.csv", dtype={"company_fkey": str})
rest["cik"] = norm(rest["company_fkey"])
for c in ("file_date", "res_begin_date", "res_end_date"): rest[c] = pd.to_datetime(rest[c], errors="coerce")

# verified actors (NAMED_CASES_PROVENANCE_20261020.md; CIKs from the pair base)
ECOLAB, TORO, ETTINGER = "31462", "737758", "jeffrey|ettinger"
K12, MUFG = "1157408", "1011659"
T_ECO = pd.Timestamp("2017-05-04"); T_K12 = pd.Timestamp("2017-04-13")
SNAP = pd.Timestamp("2017-01-01")
Y5 = pd.Timedelta(days=int(365.25 * 5)); Y3 = pd.Timedelta(days=3 * 365)

log = {}

# ---------------------------------------------------------------------------------------------------------------------
# Figure K6: design schematic on the Toro -> Ecolab appointment
# ---------------------------------------------------------------------------------------------------------------------
toro_seat = spells[(spells.person == ETTINGER) & (spells.company == TORO)].sort_values("start").iloc[0]
eco_seat = spells[(spells.person == ETTINGER) & (spells.company == ECOLAB)].sort_values("start").iloc[0]
assert eco_seat["start"] == T_ECO, eco_seat["start"]
toro_r = rest[(rest.cik == TORO) & (rest.file_date > T_ECO)].sort_values("file_date").iloc[0]          # filed 2017-12-22
eco_prior = rest[(rest.cik == ECOLAB) & (rest.file_date < T_ECO)].sort_values("file_date")               # 2003, 2007, 2012, 2015
eco_post = rest[(rest.cik == ECOLAB) & (rest.file_date > T_ECO)].sort_values("file_date").iloc[0]       # filed 2019-03-01
log["K6_dates"] = {"toro_seat_start": str(toro_seat["start"].date()), "ecolab_appointment": str(T_ECO.date()),
                   "toro_misstated": [str(toro_r.res_begin_date.date()), str(toro_r.res_end_date.date())], "toro_filed": str(toro_r.file_date.date()),
                   "ecolab_prior_filings": [str(d.date()) for d in eco_prior.file_date], "ecolab_prior_periods_begin": [str(d.date()) for d in eco_prior.res_begin_date],
                   "ecolab_post_misstated": [str(eco_post.res_begin_date.date()), str(eco_post.res_end_date.date())], "ecolab_post_filed": str(eco_post.file_date.date()),
                   "seats_end": [str(toro_seat["end"].date()) if pd.notna(toro_seat["end"]) else None, str(eco_seat["end"].date()) if pd.notna(eco_seat["end"]) else None]}

def yr(d): return d.year + (d.dayofyear - 1) / 365.25
fig, ax = plt.subplots(figsize=(11.5, 5.4))
t = yr(T_ECO)
# windows
ax.axvspan(t - 3, t, color="#dddddd", alpha=0.6, lw=0); ax.axvspan(t, t + 3, color="#f3d9c2", alpha=0.6, lw=0)
ax.text(t - 1.5, 3.75, "pre-formation window\n[$t-3$y, $t$)", ha="center", va="center", fontsize=9)
ax.text(t + 1.5, 3.75, "post-formation window\n[$t$, $t+3$y)", ha="center", va="center", fontsize=9)
ax.axvline(t, color="black", lw=1.2); ax.text(t, 4.45, "formation $t$: appointment to Ecolab, 4 May 2017", ha="center", fontsize=9.5)
# seats
y_toro, y_eco, y_rest_o, y_rest_d = 3.0, 2.3, 1.3, 0.5
end_toro = yr(toro_seat["end"]) if pd.notna(toro_seat["end"]) else 2026.0
end_eco = yr(eco_seat["end"]) if pd.notna(eco_seat["end"]) else 2026.0
ax.plot([yr(toro_seat["start"]), end_toro], [y_toro, y_toro], lw=6, color="#4c72b0", solid_capstyle="butt")
ax.text(yr(toro_seat["start"]) + 0.05, y_toro + 0.22, "director's seat at the origin (Toro): 1 Jul 2010 to 23 May 2023", ha="left", va="bottom", fontsize=8.5)
ax.plot([yr(eco_seat["start"]), end_eco], [y_eco, y_eco], lw=6, color="#dd8452", solid_capstyle="butt")
ax.text(end_eco - 0.05, y_eco - 0.22, "seat at the destination (Ecolab): 4 May 2017 to 23 May 2023", ha="right", va="top", fontsize=8.5)
ax.annotate("", xy=(t, y_eco + 0.12), xytext=(t, y_toro - 0.12), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(t + 0.1, (y_toro + y_eco) / 2, "interlock lifecycle $[t, \\min(e_o, e_d))$", fontsize=8.5, va="center")
# origin restatement: misstated period and filing
b, e, f = yr(toro_r.res_begin_date), yr(toro_r.res_end_date), yr(toro_r.file_date)
ax.plot([b, e], [y_rest_o, y_rest_o], lw=6, color="#c44e52", solid_capstyle="butt")
ax.plot([f], [y_rest_o], marker="v", ms=9, color="#c44e52", ls="none")
ax.text(b - 0.1, y_rest_o, "origin misstated period\n$[b_r, e_r]$, Nov 2015--Oct 2016", ha="right", va="center", fontsize=8.5)
ax.text(f + 0.1, y_rest_o, "origin files $f_r$\n22 Dec 2017: $e_r<t<f_r$,\nunrevealed-source exposure", ha="left", va="center", fontsize=8.5)
# destination restatements: prior filings (in pre window) and post onset
for d in eco_prior.file_date:
    if yr(d) > t - 6.5: ax.plot([yr(d)], [y_rest_d], marker="v", ms=8, color="#8172b2", ls="none")
for b2, e2 in zip(eco_prior.res_begin_date, eco_prior.res_end_date):
    if pd.notna(b2) and yr(e2) > t - 6.5: ax.plot([yr(b2), yr(e2)], [y_rest_d, y_rest_d], lw=6, color="#8172b2", alpha=0.6, solid_capstyle="butt")
ax.plot([yr(eco_post.res_begin_date), yr(eco_post.res_end_date)], [y_rest_d, y_rest_d], lw=6, color="#8172b2", solid_capstyle="butt")
ax.plot([yr(eco_post.file_date)], [y_rest_d], marker="v", ms=8, color="#8172b2", ls="none")
ax.text(yr(eco_post.res_begin_date) + 0.1, y_rest_d - 0.3, "destination onset $b_r$ = 1 Jan 2018 (post window);\nfiled 1 Mar 2019", fontsize=8.5, va="top", ha="left")
ax.text(2009.6, y_rest_d - 0.3, "destination's earlier restatements: filed 2 Aug 2012 and 27 Feb 2015\n(the 2015 filing falls in the pre window; the periods began 2011 and 2012,\nso no misstated period begins in the pre window)", fontsize=8.5, va="top", ha="left")
ax.set_xlim(2009.4, 2024.3); ax.set_ylim(-1.0, 4.8); ax.set_yticks([]); ax.set_xlabel("Calendar year")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.text(2024.2, 4.6, "bars = board seats and misstated periods\ntriangles = restatement filing dates", fontsize=8, color="#444444", ha="right", va="top")
plt.tight_layout(); plt.savefig(FIG / "figK6_design.pdf"); plt.close(); print("wrote figK6_design.pdf")

# ---------------------------------------------------------------------------------------------------------------------
# Figure K7: network snapshot and two ego networks
# ---------------------------------------------------------------------------------------------------------------------
def alive(date):
    a = edges[(edges.edge_birth <= date) & ((edges.edge_death.isna()) | (edges.edge_death > date) | (edges.ongoing == 1))]
    G = nx.Graph()
    for ca, cb, o in zip(a.company_a, a.company_b, a.origin): G.add_edge(ca, cb)
    return G, a

def recent_restater(cik, date):
    r = rest[rest.cik == cik]; return bool(((r.file_date >= date - Y5) & (r.file_date < date)).any())
def latent_at(cik, date):
    r = rest[rest.cik == cik]; return bool(((r.res_begin_date <= date) & (r.file_date > date)).any())

G, a = alive(SNAP)
lcc = max(nx.connected_components(G), key=len); H = G.subgraph(lcc).copy()
top = [n for n, _ in sorted(H.degree(), key=lambda x: x[1], reverse=True)[:400]]
Hs = H.subgraph(top).copy()
log["K7_snapshot"] = {"date": str(SNAP.date()), "edges_alive": int(a.shape[0]), "firms_alive": int(G.number_of_nodes()), "largest_component_firms": int(H.number_of_nodes()),
                      "drawn_firms": int(Hs.number_of_nodes()), "drawn_edges": int(Hs.number_of_edges()),
                      "drawn_recent_restaters": int(sum(recent_restater(n, SNAP) for n in Hs.nodes())),
                      "component_recent_restaters_share": round(float(np.mean([recent_restater(n, SNAP) for n in H.nodes()])), 4)}

def ego_panel(ax, focal, date, label_focal, origin=None, label_origin=None, title=""):
    G_, a_ = alive(date)
    if focal not in G_: ax.set_title(title + " (not in network)"); ax.axis("off"); return {}
    one = set(G_.neighbors(focal)); two = set()
    for n in one: two |= set(G_.neighbors(n))
    two -= one | {focal}
    nodes = {focal} | one | two
    E = G_.subgraph(nodes).copy()
    pos = nx.kamada_kawai_layout(E) if E.number_of_nodes() < 400 else nx.spring_layout(E, seed=7)
    col = {}
    for n in E.nodes():
        col[n] = "#c44e52" if recent_restater(n, date) else ("#dd8452" if latent_at(n, date) else "#9aa5b1")
    sizes = [260 if n == focal else (120 if n in one else 36) for n in E.nodes()]
    nx.draw_networkx_edges(E, pos, alpha=0.25, width=0.6, ax=ax)
    nx.draw_networkx_nodes(E, pos, node_color=[col[n] for n in E.nodes()], node_size=sizes, linewidths=0.4, edgecolors="black", ax=ax)
    # appointment-direction arrows for the focal firm's own ties (origin -> focal where focal is the destination)
    own = a_[((a_.company_a == focal) | (a_.company_b == focal))]
    for ca, cb, o in zip(own.company_a, own.company_b, own.origin):
        other = cb if ca == focal else ca
        if o == other:   # focal appointed the director from `other`
            ax.add_patch(FancyArrowPatch(pos[other], pos[focal], arrowstyle="-|>", mutation_scale=12, lw=1.0, color="black", shrinkA=6, shrinkB=10))
    ax.annotate(label_focal, pos[focal], xytext=(9, 9), textcoords="offset points", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if origin and origin in pos: ax.annotate(label_origin, pos[origin], xytext=(8, -12), textcoords="offset points", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=9.5); ax.axis("off")
    return {"date": str(date.date()), "one_step": len(one), "two_step": len(two),
            "one_step_recent": int(sum(recent_restater(n, date) for n in one)), "one_step_latent": int(sum(latent_at(n, date) and not recent_restater(n, date) for n in one)),
            "two_step_recent": int(sum(recent_restater(n, date) for n in two)), "two_step_latent": int(sum(latent_at(n, date) and not recent_restater(n, date) for n in two))}

fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])
pos = nx.spring_layout(Hs, seed=42, k=0.45)
deg = dict(Hs.degree())
colors = ["#c44e52" if recent_restater(n, SNAP) else "#9aa5b1" for n in Hs.nodes()]
nx.draw_networkx_edges(Hs, pos, alpha=0.10, width=0.4, ax=ax0)
nx.draw_networkx_nodes(Hs, pos, node_color=colors, node_size=[8 + 2.2 * deg[n] for n in Hs.nodes()], linewidths=0, alpha=0.9, ax=ax0)
ax0.set_title(f"(a) Interlocks alive on 1 Jan 2017: {log['K7_snapshot']['drawn_firms']} highest-degree firms of the largest component\n"
              f"({log['K7_snapshot']['largest_component_firms']:,} firms); red = filed a restatement in the previous five years", fontsize=9.5)
ax0.axis("off")
log["K7_ego_ecolab"] = ego_panel(ax1, ECOLAB, T_ECO, "Ecolab", TORO, "Toro", "(b) Ecolab's two-step neighbourhood, 4 May 2017")
log["K7_ego_k12"] = ego_panel(ax2, K12, T_K12, "K12", MUFG, "MUFG Americas", "(c) K12's two-step neighbourhood, 13 Apr 2017")
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="none", color="#c44e52", label="filed a restatement in the previous 5 years"),
           Line2D([], [], marker="o", ls="none", color="#dd8452", label="misstated period under way, not yet filed"),
           Line2D([], [], marker="o", ls="none", color="#9aa5b1", label="neither"),
           Line2D([], [], color="black", marker=">", ls="-", label="appointment direction (origin to destination) for the focal firm's ties")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(FIG / "figK7_network.pdf"); plt.close(); print("wrote figK7_network.pdf")

json.dump(log, open(RES / "N_figures_design_network.json", "w"), indent=1)
print(json.dumps(log, indent=1))
