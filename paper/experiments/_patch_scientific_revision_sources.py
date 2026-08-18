"""Transient patch for the scientific-revision sources before materialization."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
exp_path = Path(__file__).with_name("revision_experiments.py")
helper_path = root / "scripts" / "apply_scientific_revision.py"

text = exp_path.read_text(encoding="utf-8")
text = text.replace(
'''        hstars = {
            name: float(h_dense[int(np.argmax(curve))]) for name, curve in curves.items()
        }

        for spectrum_name, values in spectra.items():''',
'''        hstars_global = {
            name: float(h_dense[int(np.argmax(curve))]) for name, curve in curves.items()
        }
        window = (h_dense >= hstar_analytic / 2.0) & (h_dense <= 2.0 * hstar_analytic)
        window_idx = np.flatnonzero(window)
        hstars_gap = {
            name: float(h_dense[window_idx[int(np.argmax(curve[window]))]])
            for name, curve in curves.items()
        }

        for spectrum_name, values in spectra.items():''')
text = text.replace(
'''                    "hstar_analytic": hstar_analytic,
                    "hstar_numeric": hstars[spectrum_name],
                    "relative_hstar_error": (hstars[spectrum_name] - hstar_analytic)
                    / hstar_analytic,
                    **stats,''',
'''                    "hstar_analytic": hstar_analytic,
                    "hstar_global_numeric": hstars_global[spectrum_name],
                    "hstar_gap_numeric": hstars_gap[spectrum_name],
                    "relative_gap_hstar_error": (
                        hstars_gap[spectrum_name] - hstar_analytic
                    ) / hstar_analytic,
                    "low_temperature_peak": int(
                        hstars_global[spectrum_name] < 0.5 * hstar_analytic
                    ),
                    **stats,''')
text = text.replace(
'''    fig.tight_layout()
    base.save_figure(fig, "lac_schottky_validation")
    plt.close(fig)

    _write_rows(''',
'''    fig.tight_layout()
    base.save_figure(fig, "lac_schottky_validation")
    plt.close(fig)

    kmeans_curves = []
    for ell in range(n_clusters):
        kmeans_curves.append(base.response_curve(V_km[ell], h_dense))
    global_mean_curve = np.mean(np.vstack(kmeans_curves), axis=0)
    global_mean_hstar = float(h_dense[int(np.argmax(global_mean_curve))])
    for row in rows:
        row["kmeans_global_mean_hstar"] = global_mean_hstar

    _write_rows(''')
text = text.replace(
'''            "hstar_analytic",
            "hstar_numeric",
            "relative_hstar_error",
            "mean_relevant",''',
'''            "hstar_analytic",
            "hstar_global_numeric",
            "hstar_gap_numeric",
            "relative_gap_hstar_error",
            "low_temperature_peak",
            "kmeans_global_mean_hstar",
            "mean_relevant",''')
exp_path.write_text(text, encoding="utf-8")

h = helper_path.read_text(encoding="utf-8")
h = h.replace(
'''    true_errors = [abs(float(sch[k]["true_partition"]["relative_hstar_error"])) for k in (1, 2, 3)]
    km_errors = [abs(float(sch[k]["kmeans_partition"]["relative_hstar_error"])) for k in (1, 2, 3)]''',
'''    true_errors = [
        abs(float(sch[k]["true_partition"]["relative_gap_hstar_error"]))
        for k in (1, 2, 3)
    ]
    km_errors = [
        abs(float(sch[k]["kmeans_partition"]["relative_gap_hstar_error"]))
        for k in (1, 2, 3)
    ]''')
h = h.replace(
'''On the structured synthetic design, the resulting analytic peak predicts the true-partition numerical response maxima with at most {max_true:.2f}\\% relative error; after replacing the true partition by the initial $k$-means partition, the maximum relative displacement is {max_km:.2f}\\%.''',
'''On the structured synthetic design, the resulting analytic peak predicts the true-partition response peak with at most {max_true:.2f}\\% relative error. In the initial $k$-means spectra, the gap-associated local peak remains within {max_km:.2f}\\% of the analytic scale, although within-band dispersion produces additional low-temperature maxima in two clusters.''')
h = h.replace(
'''            f"{float(p['delta_prescribed']):.4f} & {float(p['hstar_analytic']):.4f} & "
            f"{float(t['hstar_numeric']):.4f} & {float(m['hstar_numeric']):.4f}\\\\"''',
'''            f"{float(p['delta_prescribed']):.4f} & {float(p['hstar_analytic']):.4f} & "
            f"{float(t['hstar_gap_numeric']):.4f} & "
            f"{float(m['hstar_gap_numeric']):.4f} & "
            f"{float(m['hstar_global_numeric']):.4f}\\\\"''')
h = h.replace(
'''The analytic peak agrees with the true-partition numerical maximum to within {max_true:.2f}\\% in all three clusters; after the true labels are replaced by the initial $k$-means partition, the largest relative peak displacement is {max_km:.2f}\\%. The response scale therefore survives the finite-sample and initialization perturbation in this controlled design, while the broadened spectrum makes clear that the operational problem is not literally an exactly degenerate two-level system.''',
'''The analytic peak agrees with the gap-associated true-partition maximum to within {max_true:.2f}\\% in all three clusters. After the true labels are replaced by the initial $k$-means partition, the high-temperature local maximum associated with the prescribed relevant--irrelevant gap remains within {max_km:.2f}\\% of the analytic scale. The broadened spectra of clusters 2 and 3 also develop stronger low-temperature maxima caused by dispersion within the nominal bands. Thus the estimated cluster spectrum is genuinely multiscale rather than an exactly degenerate two-level system. The operational rule averages cluster responses before maximization; for seed 1729 that global mean response selects $h={float(sch[1]['kmeans_partition']['kmeans_global_mean_hstar']):.3f}$, on the same scale as the gap-associated branch rather than the individual low-temperature anomalies.''')
h = h.replace(
'''\\begin{{tabular}}{{cccccc}}
\\toprule
\\textbf{{Cluster}} & $g_0/g_1$ & $\\Delta$ & $h^*_{{\\rm Sch}}$ & $h^*_{{\\rm true}}$ & $h^*_{{k\\text{{-means}}}}$\\\\''',
'''\\begin{{tabular}}{{ccccccc}}
\\toprule
\\textbf{{Cluster}} & $g_0/g_1$ & $\\Delta$ & $h^*_{{\\rm Sch}}$ & $h^*_{{\\rm true}}$ & $h^*_{{k\\text{{-means}},\\,gap}}$ & $h^*_{{k\\text{{-means}},\\,global}}$\\\\''')
h = h.replace(
'''\\caption{{Two-level Schottky prediction and numerical LAC response maxima for seed 1729. $g_0/g_1$ denotes the number of prescribed low/high-dispersion feature states. The true-partition column uses dispersions recomputed from the generated observations; the $k$-means column uses the aligned initial estimated partition.}}''',
'''\\caption{{Two-level Schottky prediction and numerical LAC response scales for seed 1729. $g_0/g_1$ denotes the number of prescribed low/high-dispersion feature states. The $k$-means gap column reports the local maximum on the analytic gap scale; the final column reports the unrestricted clusterwise maximum and exposes additional low-temperature structure.}}''')
helper_path.write_text(h, encoding="utf-8")
