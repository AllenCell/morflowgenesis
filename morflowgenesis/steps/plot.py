import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.utils import resample


def target_vs_prediction_scatter_metrics(x, y, niter=200):
    feats = {}
    r2 = []
    avg_ratio = []
    dist_ratio = []

    x = x.to_numpy()
    y = y.to_numpy()

    r2full = skmetrics.r2_score(y_true=x, y_pred=y)

    avg_full = np.median(100 * (y / x - 1))

    minmax = np.percentile(x, 99) - np.percentile(x, 1)
    dist_full = np.median(100 * ((y - x) / minmax))

    for _ in range(niter):
        xr, yr, xrefr = resample(x, y, x, replace=True)
        r2.append(skmetrics.r2_score(y_true=xr, y_pred=yr))

        avg_ratio.append(np.median(100 * (yr / xr - 1)))

        minmax = np.percentile(xrefr, 99) - np.percentile(xrefr, 1)
        dist_ratio.append(np.median(100 * ((yr - xr) / minmax)))

    feats["r2score"] = r2full
    feats["r2score_l"] = r2full - np.percentile(r2, 5)
    feats["r2score_h"] = np.percentile(r2, 95) - r2full

    feats["avg_ratio"] = avg_full
    feats["avg_ratio_l"] = avg_full - np.percentile(avg_ratio, 5)
    feats["avg_ratio_h"] = np.percentile(avg_ratio, 95) - avg_full

    feats["ratio_dist"] = dist_full
    feats["ratio_dist_l"] = dist_full - np.percentile(dist_ratio, 5)
    feats["ratio_dist_h"] = np.percentile(dist_ratio, 95) - dist_full

    return x, y, feats


def target_vs_prediction_scatter_plot(
    x, y, feats, title, xlabel="Label", ylabel="Pred", cc="k", fs=25
):
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
    plt.ylabel(ylabel, fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
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


def summary_plot(feats):
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
    ax[0].set_xlabel("$R^2$", fontsize=20)
    ax[1].set_xlabel("% bias")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot(df, pred_level, label_level, destdir, xlabel, ylabel):
    all_feats = {}
    for i, name in enumerate(df.columns):
        x, y = subset_data(df.iloc[:, i], pred_level, label_level)
        x, y, feats = target_vs_prediction_scatter_metrics(x, y, niter=200)
        fig, ax = target_vs_prediction_scatter_plot(
            x, y, feats, name, xlabel=xlabel, ylabel=ylabel
        )
        fig.savefig(
            os.path.join(destdir, f"scatter_{name}_{xlabel}_vs_{ylabel}.png"), transparent=True
        )
        plt.close(fig)
        all_feats[name] = feats
    if len(df.columns) > 1:
        summary = summary_plot(all_feats)
        summary.savefig(
            os.path.join(destdir, f"{xlabel}_vs_{ylabel}_summary.png"), transparent=True
        )


def subset_data(features, pred_level, label_level):
    pred = features.xs(pred_level, level="Name")
    label = features.xs(label_level, level="Name")
    # select cellids present in both
    cellids = set(pred.index.get_level_values(0)).intersection(
        set(label.index.get_level_values(0))
    )
    return label.loc[cellids], pred.loc[cellids]


def run_plot(image_objects, tags, output_name, input_step, features, label, pred):
    features_df = pd.concat([obj.load_step(input_step) for obj in image_objects]).drop_duplicates()
    features_df = features_df[features]

    for pred_filter in pred:
        if "*" in pred_filter["segmentation_name"]:
            available_levels = features_df.index.get_level_values("Name").unique().values
            for level in available_levels:
                if re.search(pred_filter["segmentation_name"], level):
                    save_dir = (
                        image_objects[0].working_dir
                        / "run_plot"
                        / output_name
                        / level.replace("/", "_")
                    )
                    save_dir.mkdir(exist_ok=True, parents=True)
                    plot(
                        features_df,
                        level,
                        label["segmentation_name"],
                        save_dir,
                        xlabel=label["description"],
                        ylabel=f"{pred_filter['description']} {level.replace('/', '_')}",
                    )
        else:
            save_dir = (
                image_objects[0].working_dir
                / "run_plot"
                / output_name
                / pred_filter["segmentation_name"]
            )
            save_dir.mkdir(exist_ok=True, parents=True)
            plot(
                features_df,
                pred_filter["segmentation_name"],
                label["segmentation_name"],
                save_dir,
                xlabel=label["description"],
                ylabel=pred_filter["description"],
            )
