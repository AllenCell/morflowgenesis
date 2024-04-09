import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as py
import sklearn.metrics as skmetrics
from plotly.subplots import make_subplots
from sklearn.utils import resample


def get_r2(x, y, niter=200):
    r2 = []
    dist_ratio = []

    r2full = skmetrics.r2_score(y_true=x, y_pred=y)

    minmax = np.percentile(x, 99) - np.percentile(x, 1)
    dist_full = np.median(100 * ((y - x) / minmax))

    for _ in range(niter):
        xr, yr, xrefr = resample(x, y, x, replace=True)
        r2.append(skmetrics.r2_score(y_true=xr, y_pred=yr))

        minmax = np.percentile(xrefr, 99) - np.percentile(xrefr, 1)
        dist_ratio.append(np.median(100 * ((yr - xr) / minmax)))

    return {
        "r2score": r2full,
        "r2score_l": r2full - np.percentile(r2, 5),
        "r2score_h": np.percentile(r2, 95) - r2full,
        "ratio_dist": dist_full,
        "ratio_dist_l": dist_full - np.percentile(dist_ratio, 5),
        "ratio_dist_h": np.percentile(dist_ratio, 95) - dist_full,
    }


def make_summary_plot(fig, pred, label, features, col=1):
    feats = {f: get_r2(pred[f], label[f]) for f in features}
    names = list(reversed(list(feats.keys())))
    for i, n in enumerate(names):
        f = feats[n]
        fig.add_trace(
            go.Bar(
                y=[i],
                x=[f["r2score"]],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[f["r2score_h"]],
                    arrayminus=[f["r2score_l"]],
                ),
                orientation="h",
                name=n,
                marker=dict(color="rgb(130, 122, 163)"),
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                y=[i],
                x=[f["ratio_dist"]],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[f["ratio_dist_h"]],
                    arrayminus=[f["ratio_dist_l"]],
                ),
                orientation="h",
                name=n,
                marker=dict(color="rgb(130, 122, 163)"),
            ),
            row=2,
            col=col,
        )
    if col == 1:
        fig.update_yaxes(
            tickangle=-45,
            tickvals=list(range(len(names))),
            ticktext=names,
            tickfont=dict(size=16),
            title_text="R<sup>2</sup>",
            row=1,
            col=col,
        )
        fig.update_yaxes(
            tickangle=-45,
            tickvals=list(range(len(names))),
            ticktext=names,
            tickfont=dict(size=16),
            title_text="% bias",
            row=2,
            col=col,
        )

    # Add bar values as ytick labels on the opposite side
    for i, n in enumerate(names):
        fig.add_annotation(
            x=1.1,
            y=i - 0.18,
            text=round(feats[n]["r2score"], 3),
            row=1,
            col=col,
            font=dict(size=16),
        )
        fig.add_annotation(
            x=6.1,
            y=i - 0.18,
            text=round(feats[n]["ratio_dist"], 3),
            row=2,
            col=col,
            font=dict(size=16),
        )

    # set bias x_lim
    fig.update_xaxes(range=[-6.1, 6.1], row=2, col=col)
    return fig, feats


def make_scatter_plot(fig, pred, label, features, r2_metrics, xlabel, ylabel, col=1):
    for i, feature in enumerate(features, start=3):
        fig.add_trace(
            go.Scatter(
                x=label[feature],
                y=pred[feature],
                mode="markers",
                # cellids
                hovertext=pred.index.get_level_values(0),
                marker=dict(color="rgb(130, 122, 163)"),
            ),
            row=i,
            col=col,
        )

        # Create a y=x line for each feature
        fig.add_trace(
            go.Scatter(
                x=[pred[feature].min(), pred[feature].max()],
                y=[pred[feature].min(), pred[feature].max()],
                mode="lines",
                line=dict(color="rgba(255, 0, 0, 0.5)", width=2),
                showlegend=False,
            ),
            row=i,
            col=col,
        )
        fig.update_yaxes(title_text=ylabel, row=i, col=col)
        fig.update_xaxes(title_text=xlabel, row=i, col=col)
        # Add text annotation to subplot with feats[features] r2 score and ratio_dist
        fig.add_annotation(
            x=label[feature].median(),
            y=pred[feature].max(),
            text=f"R<sup>2</sup>: {round(r2_metrics[feature]['r2score'], 3)}\n% bias: {round(r2_metrics[feature]['ratio_dist'], 3)}\n n={len(label)}",
            showarrow=False,
            font=dict(size=16),
            row=i,
            col=col,
        )
    return fig


def plot(features_df, groups, gt_label, features, xlabel, ylabels, save_path):
    # top row is group names above each column, bias plots have no title, then first col in each row is feature name
    titles = groups + [""] * len(groups)
    for feat in features:
        titles += [feat] * len(groups)

    fig = make_subplots(
        rows=len(features) + 2, cols=len(groups), subplot_titles=titles, vertical_spacing=0.03
    )
    for col, group in enumerate(groups, start=1):
        pred = features_df.xs(group, level="Name")
        label = features_df.xs(gt_label, level="Name")

        fig, r2_metrics = make_summary_plot(fig, pred, label, features, col=col)
        fig = make_scatter_plot(
            fig, pred, label, features, r2_metrics, xlabel, ylabels[col - 1], col=col
        )

    fig.update_layout(
        autosize=True,
        width=500 * len(groups),
        height=500 * (len(features) + 2),
        showlegend=False,
    )
    py.write_html(fig, save_path)


# def subset_data(features, pred_level, label_level):
#     pred = features.xs(pred_level, level="Name")
#     label = features.xs(label_level, level="Name")
#     # select cellids present in both
#     cellids = set(pred.index.get_level_values(0)).intersection(
#         set(label.index.get_level_values(0))
#     )
#     return label.loc[cellids], pred.loc[cellids]


def run_plot(image_objects, tags, output_name, input_step, features, label, pred):
    features_df = pd.concat([obj.load_step(input_step) for obj in image_objects]).drop_duplicates()
    destdir = image_objects[0].working_dir / "run_plot"
    destdir.mkdir(exist_ok=True, parents=True)

    groups = []
    ylabels = []
    for pred_filter in pred:
        if "*" in pred_filter["segmentation_name"]:
            available_levels = features_df.index.get_level_values("Name").unique().values
            for level in available_levels:
                if re.search(pred_filter["segmentation_name"], level):
                    groups.append(level)
                    ylabels.append(pred_filter["description"])
        else:
            groups.append(pred_filter["segmentation_name"])
            ylabels.append(pred_filter["description"])

    plot(
        features_df,
        groups=groups,
        gt_label=label["segmentation_name"],
        features=features,
        xlabel=label["description"],
        ylabels=ylabels,
        save_path=destdir / f"{output_name}.html",
    )
