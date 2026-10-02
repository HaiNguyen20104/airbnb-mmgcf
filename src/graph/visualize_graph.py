"""
visualize_graph.py
==================
Tạo các biểu đồ và giao diện trực quan hóa Đồ thị 2 phía Người dùng - Chỗ ở:
1. bipartite_analysis.png:
   - Panel A: Đồ thị 2 phía cục bộ (Sub-graph Bipartite layout) cho top users & listings
   - Panel B: Cấu trúc khối ma trận kề (Spy plot) của Bipartite Graph A = [0 R; R^T 0]
   - Panel C: Phân phối bậc của User (User Degree Distribution)
   - Panel D: Phân phối bậc của Chỗ ở (Listing/Item Degree Distribution)
2. bipartite_interactive.html:
   - File HTML trực quan kéo thả tương tác trên trình duyệt (sử dụng vis-network)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
GRAPH_DIR = DATA_DIR / "graph"
VIZ_DIR = BASE_DIR / "reports" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = GRAPH_DIR / "train_interactions.parquet"
NORM_ADJ_PATH = GRAPH_DIR / "norm_adj.pt"
META_PATH = GRAPH_DIR / "graph_meta.json"


def plot_static_figures():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_df = pd.read_parquet(TRAIN_PATH)
    num_users = meta["num_users"]
    num_items = meta["num_items"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # ── Panel A: Sub-graph 2 phía ─────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_title("A. Bipartite Subgraph (Sample Users & Listings)", fontsize=12, fontweight="bold")

    # Chọn top 5 users tương tác nhiều nhất và các items liên quan
    top_users = train_df["user_idx"].value_counts().head(5).index.tolist()
    sub_df = train_df[train_df["user_idx"].isin(top_users)]
    # Giới hạn tối đa 20 items để hình vẽ thông thoáng
    top_items = sub_df["item_idx"].value_counts().head(18).index.tolist()
    sub_df = sub_df[sub_df["item_idx"].isin(top_items)]

    B = nx.Graph()
    u_nodes = [f"U_{u}" for u in sub_df["user_idx"].unique()]
    i_nodes = [f"I_{i}" for i in sub_df["item_idx"].unique()]
    B.add_nodes_from(u_nodes, bipartite=0)
    B.add_nodes_from(i_nodes, bipartite=1)

    for _, row in sub_df.iterrows():
        B.add_edge(f"U_{row['user_idx']}", f"I_{row['item_idx']}")

    # Bipartite layout: Users bên trái (x=0), Items bên phải (x=1)
    pos = {}
    for idx, u in enumerate(u_nodes):
        pos[u] = np.array([0.0, np.linspace(0.1, 0.9, len(u_nodes))[idx]])
    for idx, i in enumerate(i_nodes):
        pos[i] = np.array([1.0, np.linspace(0.05, 0.95, len(i_nodes))[idx]])

    nx.draw_networkx_nodes(B, pos, nodelist=u_nodes, node_color="#2b5c8f", node_size=600, label="Users", ax=ax1)
    nx.draw_networkx_nodes(B, pos, nodelist=i_nodes, node_color="#e27c38", node_size=350, label="Listings", ax=ax1)
    nx.draw_networkx_edges(B, pos, edge_color="#a0aec0", alpha=0.7, width=1.5, ax=ax1)
    nx.draw_networkx_labels(B, pos, font_size=8, font_color="white", font_family="sans-serif", ax=ax1)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_axis_off()

    # ── Panel B: Ma trận kề Spy Plot ─────────────────────────
    ax2 = axes[0, 1]
    ax2.set_title("B. Block Adjacency Matrix A = [0 R; R^T 0]", fontsize=12, fontweight="bold")
    norm_adj = torch.load(NORM_ADJ_PATH, weights_only=True).coalesce()
    indices = norm_adj.indices().numpy()
    
    # Lấy ngẫu nhiên 3,000 điểm non-zero để vẽ phân bố ma trận mà không lag
    sample_size = min(3000, indices.shape[1])
    perm = np.random.choice(indices.shape[1], sample_size, replace=False)
    sub_indices = indices[:, perm]

    ax2.scatter(sub_indices[1], sub_indices[0], s=0.8, c="#4a5568", alpha=0.5)
    ax2.axvline(x=num_users, color="#e53e3e", linestyle="--", linewidth=1.2, label=f"User/Item split ({num_users})")
    ax2.axhline(y=num_users, color="#e53e3e", linestyle="--", linewidth=1.2)
    ax2.text(num_users * 0.3, num_users * 0.5, "0 (User-User)", color="#718096", fontsize=10, ha="center")
    ax2.text(num_users + (num_items * 0.4), num_users * 0.5, "R (User-Item)", color="#2b6cb0", fontsize=10, ha="center", fontweight="bold")
    ax2.text(num_users * 0.3, num_users + (num_items * 0.5), "R^T (Item-User)", color="#c05621", fontsize=10, ha="center", fontweight="bold")
    ax2.text(num_users + (num_items * 0.4), num_users + (num_items * 0.5), "0 (Item-Item)", color="#718096", fontsize=10, ha="center")
    ax2.set_xlim(0, num_users + num_items)
    ax2.set_ylim(num_users + num_items, 0)  # Matrix row orientation (0 on top)
    ax2.set_xlabel("Node Index (Global)")
    ax2.set_ylabel("Node Index (Global)")
    ax2.legend(loc="upper right", fontsize=8)

    # ── Panel C: Phân phối bậc User ──────────────────────────
    ax3 = axes[1, 0]
    ax3.set_title("C. User Degree Distribution (Interactions / User)", fontsize=12, fontweight="bold")
    user_deg = train_df["user_idx"].value_counts()
    ax3.hist(user_deg, bins=30, color="#3182ce", edgecolor="white", alpha=0.85)
    ax3.set_yscale("log")
    ax3.set_xlabel("Degree (Số phòng đã tương tác)")
    ax3.set_ylabel("Count (User count - log scale)")
    ax3.grid(axis="y", linestyle=":", alpha=0.7)

    # ── Panel D: Phân phối bậc Item ──────────────────────────
    ax4 = axes[1, 1]
    ax4.set_title("D. Listing Degree Distribution (Popularity / Item)", fontsize=12, fontweight="bold")
    item_deg = train_df["item_idx"].value_counts()
    ax4.hist(item_deg, bins=30, color="#dd6b20", edgecolor="white", alpha=0.85)
    ax4.set_yscale("log")
    ax4.set_xlabel("Degree (Số user đã đặt)")
    ax4.set_ylabel("Count (Item count - log scale)")
    ax4.grid(axis="y", linestyle=":", alpha=0.7)

    out_img = VIZ_DIR / "bipartite_analysis.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Static plot saved: {out_img}")


def generate_interactive_html():
    train_df = pd.read_parquet(TRAIN_PATH)

    # Lấy 8 users tương tác tích cực nhất và 25 listings liên quan để tạo đồ thị tương tác
    sample_users = train_df["user_idx"].value_counts().head(8).index.tolist()
    sub_df = train_df[train_df["user_idx"].isin(sample_users)]
    sample_items = sub_df["item_idx"].value_counts().head(25).index.tolist()
    sub_df = sub_df[sub_df["item_idx"].isin(sample_items)]

    nodes = []
    u_set = set(sub_df["user_idx"])
    i_set = set(sub_df["item_idx"])

    for u in u_set:
        deg = int((sub_df["user_idx"] == u).sum())
        nodes.append({
            "id": f"u_{u}",
            "label": f"User {u}",
            "group": "user",
            "title": f"Người dùng: User_{u}<br>Số tương tác trong mẫu: {deg}",
            "value": deg * 3 + 10,
            "color": "#3182ce"
        })

    for i in i_set:
        deg = int((sub_df["item_idx"] == i).sum())
        nodes.append({
            "id": f"i_{i}",
            "label": f"Phòng {i}",
            "group": "listing",
            "title": f"Chỗ ở: Listing_{i}<br>Số khách đã đặt trong mẫu: {deg}",
            "value": deg * 3 + 8,
            "color": "#dd6b20"
        })

    edges = []
    for _, row in sub_df.iterrows():
        edges.append({
            "from": f"u_{row['user_idx']}",
            "to": f"i_{row['item_idx']}",
            "color": {"color": "#a0aec0", "opacity": 0.6}
        })

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Trực quan hóa Đồ thị 2 phía Người dùng - Chỗ ở (User-Item Bipartite Graph)</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #f8fafc;
    }}
    #header {{
      padding: 16px 24px;
      background: #1e293b;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    h1 {{
      font-size: 1.25rem;
      margin: 0;
      color: #38bdf8;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.8rem;
      margin-left: 8px;
    }}
    .user-badge {{ background: #1d4ed8; color: white; }}
    .item-badge {{ background: #c2410c; color: white; }}
    #network {{
      width: 100vw;
      height: calc(100vh - 75px);
    }}
    #legend {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(30, 41, 59, 0.9);
      padding: 12px 18px;
      border-radius: 8px;
      border: 1px solid #475569;
      font-size: 0.85rem;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .dot {{
      height: 12px;
      width: 12px;
      display: inline-block;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
    }}
  </style>
</head>
<body>
  <div id="header">
    <div>
      <h1>Trực quan hóa Đồ thị 2 phía MMGCF (User–Item Bipartite Graph)</h1>
      <span style="font-size: 0.85rem; color: #94a3b8;">Bạn có thể dùng chuột kéo thả các node, cuộn chuột để Zoom in/out</span>
    </div>
    <div>
      <span class="badge user-badge">Người dùng (Users): {len(u_set)}</span>
      <span class="badge item-badge">Chỗ ở (Listings): {len(i_set)}</span>
      <span class="badge" style="background:#475569">Cạnh liên kết: {len(edges)}</span>
    </div>
  </div>

  <div id="network"></div>

  <div id="legend">
    <div style="margin-bottom: 6px;"><span class="dot" style="background: #3182ce;"></span> <b>Node Xanh:</b> Người dùng (User)</div>
    <div><span class="dot" style="background: #dd6b20;"></span> <b>Node Cam:</b> Chỗ ở (Listing / Room)</div>
  </div>

  <script type="text/javascript">
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});

    const container = document.getElementById("network");
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: "dot",
        scaling: {{ min: 14, max: 32 }},
        font: {{ color: "#f8fafc", size: 13, face: "sans-serif" }}
      }},
      edges: {{
        width: 1.5,
        smooth: {{ type: "continuous" }}
      }},
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -70,
          centralGravity: 0.015,
          springLength: 90,
          springConstant: 0.08
        }},
        maxVelocity: 50,
        minVelocity: 0.1,
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100
      }}
    }};

    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>"""

    out_html = VIZ_DIR / "bipartite_interactive.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Interactive HTML saved: {out_html}")


if __name__ == "__main__":
    plot_static_figures()
    generate_interactive_html()
