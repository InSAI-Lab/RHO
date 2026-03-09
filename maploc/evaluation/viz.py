# Copyright (c) Meta Platforms, Inc. and affiliates.

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..osm.viz import Colormap, plot_nodes
from ..utils.io import write_torch_image
from ..utils.viz_2d import features_to_RGB, plot_images, save_plot
from ..utils.viz_localization import (
    add_circle_inset,
    likelihood_overlay,
    plot_dense_rotations,
    plot_pose,
)

def angle_mid(yaw_1, yaw_2):
    # yaw between [-180, 180]
    diff = yaw_2 - yaw_1
    diff = (diff + 180) % 360 - 180 # Normalize diff to [-180, 180]
    mid = yaw_1 + diff / 2
    mid = (mid + 180) % 360 - 180 # Normalize mid to [-180, 180]
    return mid

def plot_example_views(
    idx,
    data,
    view_idx=None,
    out_dir=None,
):
    scene, name, rasters, uv_gt = (data[k] for k in ("scene", "name", "map", "uv"))
    image = data["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)
    map_viz = Colormap.apply(rasters)
    plot_images(
        [image, map_viz],
        titles=[f"View {view_idx}", "map"],
        dpi=75,
        cmaps="jet",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        p = str(out_dir / f"pano_{idx:04d}_{scene}_{name_}_{{}}.pdf")
        save_plot(p.format(f"View{view_idx}_image"))
        plt.close()
    return

def plot_example_single(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    scene, name, rasters, uv_gt = (data[k] for k in ("scene", "name", "map", "uv"))
    uv_gps = data.get("uv_gps")
    yaw_gt = data["roll_pitch_yaw"][-1].numpy()
    image = data["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)

    lp_uvt = lp_uv = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_uvt = lp_uv = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_uvt = lp_uv = pred["scores_unmasked"]
    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")
    if show_fused and "uv_fused" in pred:
        uv_p, yaw_p = pred["uv_fused"], pred.get("yaw_fused")
    feats_map = pred["map"]["map_features"][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    # Plot prediction images
    plot_images(
        [image, map_viz, overlay, feats_map_rgb],
        titles=[text1, "map", "likelihood", "neural map"],
        dpi=75,
        cmaps="jet",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2])

    if show_gps and uv_gps is not None:
        plot_pose([1], uv_gps, c="blue")
    plot_pose([1], uv_gt, yaw_gt, c="red")
    plot_pose([1], uv_p, yaw_p, c="k")
    plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[2], inset_center)
    axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        p = str(out_dir / f"{scene}_{name_}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], uv_gt, yaw_gt, c="red")
            plot_pose([0], pred["uv_max"], pred["yaw_max"], c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_uv], cmaps="jet")
            plot_dense_rotations(0, lp_uvt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )
            write_torch_image(p.format("image").replace("pdf", "jpg"), image.numpy())

    if not plot_bev:
        return

    feats_q = pred["features_bev"]
    mask_bev = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_q, dim=0)
    conf_q = pred["confidence_bev"]
    conf_q = conf_q.masked_fill(~mask_bev, np.nan)
    (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
    # feats_map_rgb, feats_q_rgb, = features_to_RGB(
    #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
    norm_map = torch.norm(feats_map, dim=0)

    plot_images(
        [conf_q, feats_q_rgb, norm_map] + ([] if prior is None else [prior]),
        titles=["BEV confidence", "BEV features", "map norm"]
        + ([] if prior is None else ["map prior"]),
        dpi=50,
        cmaps="jet",
    )
    plt.show()

    if out_dir is not None:
        save_plot(p.format("bev"))
        plt.close()

def plot_example_paired(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    """
    Visualizes a single prediction, adapted for panorama processing.
    'pred' contains the final prediction for one panorama.
    'data' contains the ground truth for the main view (view 1) of that panorama.
    """
    gt_data = data[1]
    scene, name, rasters, uv_gt = (gt_data[k] for k in ("scene", "name", "map", "uv"))
    uv_gps = gt_data.get("uv_gps")
    yaw_gt_v2 = gt_data["roll_pitch_yaw"][-1].numpy()
    yaw_gt_v1 = data[0]["roll_pitch_yaw"][-1].numpy()
    yaw_gt = angle_mid(yaw_gt_v1, yaw_gt_v2)
    rasters_v1 = data[0]["map"]
    image = gt_data["image"].permute(1, 2, 0)
    image_v1 = data[0]["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~gt_data["valid"].unsqueeze(-1), 0.3)
        image_v1 = image_v1.masked_fill(~data[0]["valid"].unsqueeze(-1), 0.3)

    lp_uvt = lp_uv = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_uvt = lp_uv = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_uvt = lp_uv = pred["scores_unmasked"]
    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")
    if show_fused and "uv_fused" in pred:
        uv_p, yaw_p = pred["uv_fused"], pred.get("yaw_fused")
    
    # Ensure map features are on CPU and in numpy format for visualization
    feats_map = pred["map"]["map_features"][0][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lat/lon={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    map_viz_v1 = Colormap.apply(rasters_v1)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    plot_images(
        [image, map_viz, overlay, feats_map_rgb],
        titles=[f"View 2 - {text1}", "map", "likelihood", "neural map"],
        dpi=75,
        cmaps="jet",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2])

    if show_gps and uv_gps is not None:
        plot_pose([1], uv_gps, c="blue")
    plot_pose([1], uv_gt, yaw_gt, c="red")
    plot_pose([1], uv_p, yaw_p, c="k")
    plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[2], inset_center)
    axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        # p = str(out_dir / f"pano_{idx:04d}_{scene}_{name_}_{{}}.pdf")
        p = str(out_dir / f"pano_{idx:04d}_{scene}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

    plot_images(
        [image, image_v1],
        titles=["View2", "View1"],
        dpi=75,
        cmaps="jet",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        save_plot(p.format(f"View_images"))
        plt.close()

    if not plot_bev:
        return

    # --- START: Visualization for Panorama BEV features ---

    # 1. Visualize the FINAL concatenated and rotated BEV feature
    feats_final = pred["features_bev"]
    mask_final = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    '''
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_final, dim=0)
    '''
    conf_q = pred["confidence_bev"]
    conf_q = conf_q.masked_fill(~mask_final, np.nan)
    (feats_final_rgb,) = features_to_RGB(feats_final.numpy(), masks=[mask_final.numpy()])
    norm_map = torch.norm(feats_map, dim=0)

    # 2. Visualize the individual BEV features BEFORE concatenation (optional, for debugging)
    if "features_bev_individual" in pred:
        feats_views = pred["features_bev_individual"]
        confidence_views = pred["confidence_bev_individual"]
        masks_views = pred["valid_bev_individual"]
        
        # Create RGB visualizations for each of the 3 views
        feats_views_rgb = features_to_RGB(*feats_views.numpy(), masks=masks_views.numpy())
        
        plot_images(
            [conf_q, feats_final_rgb, norm_map] + ([] if prior is None else [prior]),
            titles=["Pano BEV confidence", "Pano BEV features", "map norm"]
            + ([] if prior is None else ["map prior"]),
            dpi=50,
            cmaps="jet",
        )
        plt.show()
        if out_dir is not None:
            save_plot(p.format("pano_bev_features"))
            plt.close()

        plot_images(
            [feats_views_rgb[0], feats_views_rgb[1]],
            titles=["View 1", "View 2"],
            dpi=50,
            cmaps="jet",
        )
        plt.show()

        if out_dir is not None:
            save_plot(p.format("views_bev_features"))
            plt.close()

    else: # Fallback for non-panorama models
        plot_images([feats_final_rgb], titles=["BEV Features"], dpi=75)

    # --- END: Visualization for Panorama BEV features ---



def plot_example_pano(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    """
    Visualizes a single prediction, adapted for panorama processing.
    'pred' contains the final prediction for one panorama.
    'data' contains the ground truth for the main view (view 1) of that panorama.
    """
    gt_data = data[1]
    scene, name, rasters, uv_gt = (gt_data[k] for k in ("scene", "name", "map", "uv"))
    uv_gps = gt_data.get("uv_gps")
    yaw_gt = gt_data["roll_pitch_yaw"][-1].numpy()
    rasters_v1 = data[0]["map"]
    rasters_v3 = data[2]["map"]
    image = gt_data["image"].permute(1, 2, 0)
    image_v1 = data[0]["image"].permute(1, 2, 0)
    image_v3 = data[2]["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~gt_data["valid"].unsqueeze(-1), 0.3)
        image_v1 = image_v1.masked_fill(~data[0]["valid"].unsqueeze(-1), 0.3)
        image_v3 = image_v3.masked_fill(~data[2]["valid"].unsqueeze(-1), 0.3)

    lp_uvt = lp_uv = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_uvt = lp_uv = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_uvt = lp_uv = pred["scores_unmasked"]
    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")
    if show_fused and "uv_fused" in pred:
        uv_p, yaw_p = pred["uv_fused"], pred.get("yaw_fused")
    
    # Ensure map features are on CPU and in numpy format for visualization
    feats_map = pred["map"]["map_features"][0][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lat/lon={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    map_viz_v1 = Colormap.apply(rasters_v1)
    map_viz_v3 = Colormap.apply(rasters_v3)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    plot_images(
        [image, map_viz, overlay, feats_map_rgb],
        titles=[f"View 2 - {text1}", "map", "likelihood", "neural map"],
        dpi=75,
        cmaps="jet",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2])

    if show_gps and uv_gps is not None:
        plot_pose([1], uv_gps, c="blue")
    plot_pose([1], uv_gt, yaw_gt, c="red")
    plot_pose([1], uv_p, yaw_p, c="k")
    plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[2], inset_center)
    axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        # p = str(out_dir / f"pano_{idx:04d}_{scene}_{name_}_{{}}.pdf")
        p = str(out_dir / f"pano_{idx:04d}_{scene}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

    plot_images(
        [image_v1, image, image_v3],
        titles=["left", "mid", "right"],
        dpi=75,
        cmaps="jet",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        save_plot(p.format(f"View_images"))
        plt.close()

    if not plot_bev:
        return

    # --- START: Visualization for Panorama BEV features ---

    # 1. Visualize the FINAL concatenated and rotated BEV feature
    feats_final = pred["features_bev"]
    mask_final = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    '''
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_final, dim=0)
    '''
    conf_q = pred["confidence_bev"]
    conf_q = conf_q.masked_fill(~mask_final, np.nan)
    (feats_final_rgb,) = features_to_RGB(feats_final.numpy(), masks=[mask_final.numpy()])
    norm_map = torch.norm(feats_map, dim=0)

    # 2. Visualize the individual BEV features BEFORE concatenation (optional, for debugging)
    if "features_bev_individual" in pred:
        feats_views = pred["features_bev_individual"]
        confidence_views = pred["confidence_bev_individual"]
        masks_views = pred["valid_bev_individual"]
        
        # Create RGB visualizations for each of the 3 views
        feats_views_rgb = features_to_RGB(*feats_views.numpy(), masks=masks_views.numpy())
        
        plot_images(
            [conf_q, feats_final_rgb, norm_map] + ([] if prior is None else [prior]),
            titles=["Pano BEV confidence", "Pano BEV features", "map norm"]
            + ([] if prior is None else ["map prior"]),
            dpi=50,
            cmaps="jet",
        )
        plt.show()
        if out_dir is not None:
            save_plot(p.format("pano_bev_features"))
            plt.close()

        plot_images(
            [feats_views_rgb[0], feats_views_rgb[1], feats_views_rgb[2]],
            titles=["left", "mid", "right"],
            dpi=50,
            cmaps="jet",
        )
        plt.show()

        if out_dir is not None:
            save_plot(p.format("views_bev_features"))
            plt.close()

    else: # Fallback for non-panorama models
        plot_images([feats_final_rgb], titles=["BEV Features"], dpi=75)

    # --- END: Visualization for Panorama BEV features ---


def plot_example_sequential(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return
