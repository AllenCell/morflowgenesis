import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from prefect import flow
from sklearn.decomposition import PCA
from sklearn.utils import resample


def target_vs_prediction_scatter_metrics(x, y, niter=200):
    feats = {}
    r2 = []
    avg_ratio = []
    dist_ratio = []
    deviation = []
    log = np.isnan(x) | np.isnan(y)

    if len(list(x.shape)) > 1:
        log = np.sum(np.isnan(x) | np.isnan(y), axis=1) > 0
    else:
        log = np.isnan(x) | np.isnan(y)
    x0 = x[~log].to_numpy()
    y0 = y[~log].to_numpy()

    # if no xref is supplied, then just use the target data given for noamlization of percent bias
    xref0 = x0

    r2full = skmetrics.r2_score(y_true=x0, y_pred=y0)

    avg_full = np.mean((y0 / x0) - 1) * 100
    avg_full = np.median(100 * (y0 / x0 - 1))

    minmax = np.percentile(x0, 99) - np.percentile(x0, 1)
    dist_full = np.median(100 * ((y0 - x0) / minmax))

    deviation_full = np.median(y0 - x0)

    for _ in range(niter):
        xr, yr, xrefr = resample(x0, y0, xref0, replace=True)
        r2.append(skmetrics.r2_score(y_true=xr, y_pred=yr))

        avg_ratio.append(np.median(100 * (yr / xr - 1)))

        minmax = np.percentile(xrefr, 99) - np.percentile(xrefr, 1)
        dist_ratio.append(np.median(100 * ((yr - xr) / minmax)))
        deviation.append(np.median(yr - xr))

    feats["r2score"] = r2full
    feats["r2score_l"] = r2full - np.percentile(r2, 5)
    feats["r2score_h"] = np.percentile(r2, 95) - r2full

    feats["avg_ratio"] = avg_full
    feats["avg_ratio_l"] = avg_full - np.percentile(avg_ratio, 5)
    feats["avg_ratio_h"] = np.percentile(avg_ratio, 95) - avg_full

    feats["ratio_dist"] = dist_full
    feats["ratio_dist_l"] = dist_full - np.percentile(dist_ratio, 5)
    feats["ratio_dist_h"] = np.percentile(dist_ratio, 95) - dist_full

    feats["deviation"] = deviation_full
    feats["deviation_l"] = deviation_full - np.percentile(deviation, 5)
    feats["deviation_h"] = np.percentile(deviation, 95) - deviation_full

    return x0, y0, feats


def target_vs_prediction_scatter_plot(x, y, feats, title, cc="k", fs=14):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title(title)
    xlh = (np.nanmin(x), np.nanmax(x))
    plt.plot(xlh, xlh, "--", color=cc)
    plt.plot(
        x,
        y,
        marker=".",
        markerfacecolor="None",
        markeredgewidth=0.5,
        markeredgecolor=cc,
        linestyle="None",
        label="",
    )
    plt.ylabel("Pred")
    plt.xlabel("Label")
    plt.axis("equal")

    score = feats["r2score"]
    perc_bias = feats["ratio_dist"]
    plt.text(
        0.02,
        0.96,
        "%bias = " + str(np.round(perc_bias, 2)),
        va="top",
        ha="left",
        fontsize=fs * 1,
        transform=ax.transAxes,
    )
    plt.text(
        0.02,
        0.83,
        "R² = " + str(np.round(score, 2)),
        va="top",
        ha="left",
        fontsize=fs * 1,
        transform=ax.transAxes,
    )
    plt.text(
        0.02,
        0.70,
        "n = " + str(x.shape[0]),
        va="top",
        ha="left",
        fontsize=fs * 1,
        transform=ax.transAxes,
    )
    return fig, ax


def perform_PCA(x, y, n_components):
    # dimensionality reduction, 578 -> n_components
    pca = PCA(n_components=n_components)
    cols = x.columns
    x = pca.fit_transform(x)
    y.columns = cols
    y = pca.transform(y)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    new_columns = ["PC" + str(i + 1) for i in range(n_components)]
    x.columns = new_columns
    y.columns = new_columns
    return x, y


def summary_plot(feats, destdir):
    names = list(reversed(list(feats.keys())))
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    errorbar_params = {
        "capsize": 3,
        "color": (130 / 255, 122 / 255, 163 / 255),
        "linewidth": 1,
        "height": 0.8,
        "zorder": 10,
    }

    bias_xlim = (
        np.max(
            [np.abs(feats[n]["ratio_dist"] + feats[n]["ratio_dist_h"]) for n in names]
            + [np.abs(feats[n]["ratio_dist"] - feats[n]["ratio_dist_l"]) for n in names]
        )
        * 1.1
    )
    bias_xlim = 6
    for i, n in enumerate(names):
        f = feats[n]
        errorbar_params.update({"y": i})
        ax[0].barh(
            width=f["r2score"],
            xerr=np.vstack([f["r2score_l"], f["r2score_h"]]),
            **errorbar_params,
        )
        ax[0].text(
            1.05,
            i,
            str(np.round(np.nanmean(f["r2score"]), 2)),
            fontsize=14,
        )
        ax[1].barh(
            width=[f["ratio_dist"]],
            xerr=np.vstack([f["ratio_dist_l"], f["ratio_dist_h"]]),
            **errorbar_params,
        )
        ax[1].text(
            bias_xlim * 1.1,
            i,
            str(np.round(np.nanmean(f["ratio_dist"]), 2)),
            fontsize=14,
        )
    ax[1].set_xlim([-bias_xlim, bias_xlim])
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names)
    ax[0].grid(zorder=0)
    ax[1].grid(zorder=0)
    ax[0].set_xlabel("$R^2$")
    ax[1].set_xlabel("% bias")
    plt.tight_layout()
    fig.savefig(os.path.join(destdir, "PC_summary.png"), transparent=True)
    plt.close(fig)


def plot(x, y, destdir):
    all_feats = {}
    for name in x.columns:
        x0, y0, feats = target_vs_prediction_scatter_metrics(
            x[name], y[name.replace("label", "pred")], niter=200
        )
        fig, ax = target_vs_prediction_scatter_plot(x0, y0, feats, name)
        fig.savefig(os.path.join(destdir, f"scatter_{name}.png"))
        plt.close(fig)
        all_feats[name] = feats
    if len(x.columns) > 1:
        summary_plot(all_feats, destdir)


@flow(log_prints=True)
def run_plot(image_object, step_name, output_name, input_step, features, pca_n_components=10):
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object
    (image_object.working_dir / step_name / output_name).mkdir(exist_ok=True, parents=True)
    features_df = image_object.load_step(input_step)
    for feat in features:
        label = features_df[[col for col in features_df.columns if "label" in col and feat in col]]
        pred = features_df[[col for col in features_df.columns if "pred" in col and feat in col]]
        if label.shape[1] > pca_n_components:
            label, pred = perform_PCA(label, pred, pca_n_components)
        print(label.columns, pred.columns)
        plot(label, pred, image_object.working_dir / step_name / output_name)
