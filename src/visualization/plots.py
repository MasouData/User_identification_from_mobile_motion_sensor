from pathlib import Path

import matplotlib.pyplot as plt


def plot_model_comparison(
    model_summary,
    output_path=None,
):
    """
    Plot mean Macro-F1 for each evaluated model.

    Error bars represent the standard deviation across
    repeated stratified train/validation splits.

    Parameters
    ----------
    model_summary : pandas.DataFrame
        Output of summarize_model_scores().

        Expected columns:
        - model
        - mean_macro_f1
        - std_macro_f1

    output_path : str or Path, optional
        If provided, save the figure to this location.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    summary = (
        model_summary
        .sort_values(
            "mean_macro_f1",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.barh(
        summary["model"],
        summary["mean_macro_f1"],
        xerr=summary["std_macro_f1"],
        capsize=4,
    )

    ax.set_xlim(0, 1)

    ax.set_xlabel(
        "Mean Macro-F1"
    )

    ax.set_ylabel(
        "Model"
    )

    ax.set_title(
        "Model Comparison for Motion-Based User Identification"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Add score beside each bar
    for bar, score in zip(
        bars,
        summary["mean_macro_f1"],
    ):
        ax.text(
            min(score + 0.015, 0.97),
            bar.get_y()
            + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
        )

    fig.tight_layout()

    # Save figure when requested
    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

    return fig