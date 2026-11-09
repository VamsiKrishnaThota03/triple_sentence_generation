"""Draw t5_architecture.png (Figure 1 in main.tex): triplet -> T5 encoder-decoder -> sentence."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "t5_architecture.png"

BOXES = [
    # (x, y, w, h, title, body, colour)
    (0.02, 0.30, 0.19, 0.40, "Knowledge triplet", "(Albert Einstein,\ndeveloped,\ntheory of relativity)", "#e0f2fe"),
    (0.26, 0.30, 0.17, 0.40, "Linearised input", "subject: ...\npredicate: ...\nobject: ...", "#ede9fe"),
    (0.49, 0.56, 0.20, 0.32, "T5 encoder", "SentencePiece tokens\nself-attention layers", "#dcfce7"),
    (0.49, 0.12, 0.20, 0.32, "T5 decoder", "cross-attention\nbeam search (4 beams)", "#dcfce7"),
    (0.76, 0.30, 0.22, 0.40, "Sentence", "\"Albert Einstein developed\nthe theory of relativity.\"", "#fee2e2"),
]


def main():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for x, y, w, h, title, body, colour in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=colour, edgecolor="#334155", linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=10.5, weight="bold",
                color="#0f172a")
        ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=8.5,
                color="#334155", family="DejaVu Sans")

    def arrow(x0, y0, x1, y1, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                     color="#475569", linewidth=1.3))

    arrow(0.225, 0.5, 0.248, 0.5)
    arrow(0.442, 0.6, 0.478, 0.68)
    arrow(0.59, 0.545, 0.59, 0.455)
    arrow(0.702, 0.3, 0.748, 0.45)
    ax.text(0.59, 0.03, "fine-tuned on REBEL sentences (cross-entropy loss)", ha="center",
            fontsize=8.5, color="#64748b", style="italic")
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
