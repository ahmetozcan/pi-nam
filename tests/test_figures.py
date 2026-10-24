"""Stage 10 shared figure style (src/revision/plot_style.py)."""

import matplotlib.pyplot as plt

from src.revision import plot_style as ps


def test_style_fonts_readable_at_print_size():
    ps.apply()
    for key in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                "legend.fontsize", "axes.titlesize"):
        assert float(plt.rcParams[key]) >= 8
    assert plt.rcParams["pdf.fonttype"] == 42


def test_save_writes_pdf_and_png_at_text_width(tmp_path):
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH, 2))
    ax.plot([0, 1], [0, 1])
    ps.save(fig, tmp_path, "f")
    assert (tmp_path / "f.pdf").stat().st_size > 0 and (tmp_path / "f.png").stat().st_size > 0
    assert abs(ps.TEXT_WIDTH * 2.54 - 16.0) < 0.1
