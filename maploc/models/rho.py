# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth, make_pano_grid
from .map_encoder import MapEncoder
from .metrics_v1gt import AngleError, AngleRecall, Location2DError, Location2DRecall
from .voting import (
    TemplateSampler,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "fusion_strategy": "pano_prior",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "depth_loss_weight": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }


    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )
        grid_xz_pano = make_pano_grid(conf.z_max, ppm)
        self.template_sampler_pano = TemplateSampler(
            grid_xz_pano, ppm, conf.num_rotations
        )
        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)
        
        # Define the number of views a pano contains
        self.num_views = 3
        # Introduce a learnable weighted fusion scalar
        self.fusion_alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.fusion_beta = torch.nn.Parameter(torch.tensor(0.5))

    def _is_panorama_batch(self, data, num_views):
        """
        Check if the batch contains panorama views (groups of 3).
        """
        batch_size = data["image"].shape[0]
        return batch_size % num_views == 0 and batch_size >= num_views
    
    def _rotate_bev_around_camera(self, features, angle_deg):

        is_3d = features.dim() == 3
        if is_3d:
            features = features.unsqueeze(0)

        N, C, H, W = features.shape
        device = features.device
        dtype = features.dtype

        # 1. Create a bigger canvas to cover the pano features
        # MAke sure the orthocenter of the equilateral triangle is on the center of the canvas
        padding = (0, 0, H, 2*H)         # F.pad:(left, right, top, bottom)
        features_padded = F.pad(features, padding, mode='constant', value=0)
        new_H, new_W = features_padded.shape[-2:]

        angle_rad = angle_deg * np.pi / 180.0
        cos_a = torch.cos(torch.tensor(angle_rad, dtype=dtype, device=device))
        sin_a = torch.sin(torch.tensor(angle_rad, dtype=dtype, device=device))
        R = torch.tensor([[cos_a, -sin_a], [sin_a,  cos_a]], dtype=dtype, device=device)

        # The original center of the camera ((W-1)/2, H-1)
        center_pix_x = (W - 1) / 2
        center_pix_y = 2*H - 1

        # Create the rotate matrix
        T1 = torch.tensor([[1, 0, -center_pix_x], [0, 1, -center_pix_y], [0, 0, 1]], dtype=dtype, device=device)
        R_homo = torch.eye(3, dtype=dtype, device=device)
        R_homo[:2, :2] = R
        T2 = torch.tensor([[1, 0, center_pix_x], [0, 1, center_pix_y], [0, 0, 1]], dtype=dtype, device=device)
        transform_pix = T2 @ R_homo @ T1

        # Normalize the affine grid
        norm_to_pix = torch.tensor([
            [(new_W - 1) / 2, 0, (new_W - 1) / 2],
            [0, (new_H - 1) / 2, (new_H - 1) / 2],
            [0, 0, 1]
        ], dtype=dtype, device=device)
        pix_to_norm = torch.inverse(norm_to_pix)
        transform_norm = pix_to_norm @ transform_pix @ norm_to_pix
        theta = transform_norm[:2, :]

        grid = F.affine_grid(
            theta.unsqueeze(0).expand(N, 2, 3),
            size=[N, C, new_H, new_W],
            align_corners=False,
        )
        rotated_padded = F.grid_sample(
            features_padded, grid, align_corners=False, padding_mode="zeros"
        )


        return rotated_padded.squeeze(0) if is_3d else rotated_padded


    def _stitch_panorama_bev(self, f_bev, valid_bev):
        """
        Stitches BEV features and masks from 3 panorama views by rotating them
        around the camera position to form a single, coherent 360-degree BEV.

        Args:
            f_bev: [B, C, H, W] - BEV features for all 3 views.
            valid_bev: [B, H, W] - Valid masks for each view.

        Returns:
            f_bev_stitched: [B//3, C, H, W] - Stitched features for each panorama.
            valid_stitched: [B//3, H, W] - Combined valid mask for each panorama.
        """
        B, C, H, W = f_bev.shape
        device = f_bev.device
        assert B % 3 == 0, "Batch size must be a multiple of 3 for panorama processing"
        num_panoramas = B // 3

        f_bev_stitched = torch.zeros(
            (num_panoramas, C, 4*H, W), device=device, dtype=f_bev.dtype
        )
        valid_stitched = torch.zeros(
            (num_panoramas, 4*H, W), device=device, dtype=torch.bool
        )

        for i in range(num_panoramas):
            start_idx = i * 3

            f_v1, v_v1 = f_bev[start_idx], valid_bev[start_idx]
            f_v2, v_v2 = f_bev[start_idx + 1], valid_bev[start_idx + 1]
            f_v3, v_v3 = f_bev[start_idx + 2], valid_bev[start_idx + 2]
            
            # View 1 (0 deg yaw): The reference view, no rotation needed.
            f_v1_rot = self._rotate_bev_around_camera(f_v1, 0)
            v_v1_rot = (
                self._rotate_bev_around_camera(
                    v_v1.unsqueeze(0).float(), 0
                )
                .squeeze(0) > 0.5
            )
            
            # View 2 : Rotate its content by 120 deg to align.
            f_v2_rot = self._rotate_bev_around_camera(f_v2, 120)
            v_v2_rot = (
                self._rotate_bev_around_camera(
                    v_v2.unsqueeze(0).float(), 120
                )
                .squeeze(0) > 0.5
            )

            # View 3 : Rotate its content by -120 deg to align.
            f_v3_rot = self._rotate_bev_around_camera(f_v3, -120)
            v_v3_rot = (
                self._rotate_bev_around_camera(
                    v_v3.unsqueeze(0).float(), -120
                )
                .squeeze(0) > 0.5
            )

            
            # --- Robust Pasting Logic ---
            # Mask each view's features with its validity mask, setting invalid areas to 0.
            f_v1_masked = torch.where(v_v1_rot.unsqueeze(0), f_v1_rot, 0.0)
            f_v2_masked = torch.where(v_v2_rot.unsqueeze(0), f_v2_rot, 0.0)
            f_v3_masked = torch.where(v_v3_rot.unsqueeze(0), f_v3_rot, 0.0)
            
            numerator = (f_v1_masked * v_v1_rot.unsqueeze(0).float() +
                     f_v2_masked * v_v2_rot.unsqueeze(0).float() +
                     f_v3_masked * v_v3_rot.unsqueeze(0).float())
            
            denominator = (v_v1_rot.float() + v_v2_rot.float() + v_v3_rot.float())
            denominator = torch.clamp(denominator, min=0.5)

            f_bev_stitched[i] = numerator / denominator.unsqueeze(0)

            # Since the masks should now be perfectly disjoint, we can simply add the masked features.
            #f_bev_stitched[i] = f_v1_masked + f_v2_masked + f_v3_masked
            valid_stitched[i] = v_v1_rot | v_v2_rot | v_v3_rot

        return f_bev_stitched, valid_stitched

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None, is_panorama=True):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        if is_panorama:
            templates = self.template_sampler_pano(f_bev)
        else:
            templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float().unsqueeze(1)) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        # Check if we have panorama views and process accordingly
        is_panorama = self._is_panorama_batch(data, num_views=self.num_views)
        confidence_bev = pred_bev.get("confidence")
        
        if is_panorama and self.conf.fusion_strategy == "view1_yaw_prior":
            # For panorama batches, select map data corresponding to the main view (view1)
            f_map = f_map[::3]
            if "log_prior" in pred_map:
                pred_map["log_prior"][0] = pred_map["log_prior"][0][::3]
            
            # Store individual view features for debugging before stitching
            features_bev_individual = f_bev.clone()
            valid_bev_individual = valid_bev.clone()
            if confidence_bev is not None:
                confidence_bev_individual = confidence_bev.clone()
            
            # Stitch the 3 views into a full panorama
            f_bev_stitched, valid_bev_stitched = self._stitch_panorama_bev(
                f_bev, valid_bev
            )
            
            confidence_bev_stitched = None
            if confidence_bev is not None:
                # Also stitch confidence maps (they are like single-channel features)
                confidence_bev_stitched, _ = self._stitch_panorama_bev(
                    confidence_bev.unsqueeze(1), valid_bev
                )
                confidence_bev_stitched = confidence_bev_stitched.squeeze(1)

            f_bev_final, valid_bev_final, confidence_bev_final = f_bev_stitched, valid_bev_stitched, confidence_bev_stitched
        else:
            # Standard single-view processing
            f_bev_final, valid_bev_final, confidence_bev_final = f_bev, valid_bev, confidence_bev

        # --- Stage 1: Pano mode for robust (x,y) prior ---
        # Use the final BEV features for exhaustive voting
        scores_pano = self.exhaustive_voting(
            f_bev_final, f_map, valid_bev_final, confidence_bev_final, is_panorama=True
        ) # Shape: [B/3, N_rot, H, W]

        # Create a spatial prior (xy_heatmap) by taking the LogSumExp over the rotation dimension
        # This heatmap has high values where the location is likely, regardless of yaw.
        log_probs_pano = log_softmax_spatial(scores_pano.permute(0, 2, 3, 1)) # Shape: [B/3, H, W]
        xy_log_heatmap = torch.logsumexp(log_probs_pano, dim=-1)

        # --- Stage 2: Single-view mode for precise yaw, guided by the (x,y) prior ---
        f_bev_v1 = features_bev_individual[::3]
        valid_bev_v1 = valid_bev_individual[::3]
        confidence_bev_v1 = confidence_bev_individual[::3] if confidence_bev_individual is not None else None

        scores_view1 = self.exhaustive_voting(
            f_bev_v1, f_map, valid_bev_v1, confidence_bev_v1, is_panorama=False
        ) # Shape: [B/3, N_rot, H, W]
        log_probs_view1 = log_softmax_spatial(scores_view1.permute(0, 2, 3, 1)) # Shape: [B/3, H, W, N_rot]
        # Marginalize out the spatial dimensions (H and W) to get yaw distribution
        # Flatten H and W dims: [B/3, H*W, N_rot]
        # B_pano, H, W, N_rot = log_probs_view1.shape
        # view1_flat_spatial = log_probs_view1.view(B_pano, H * W, N_rot)
        # logsumexp over the spatial dimension to get yaw prior
        # yaw_log_heatmap = torch.logsumexp(view1_flat_spatial, dim=1) # Shape: [B/3, N_rot]


        # --- Stage 3: Fuse the sharp yaw scores with the robust xy prior.
        scores_v1_corrected = (1 - self.fusion_alpha) * log_probs_view1 + self.fusion_alpha * xy_log_heatmap.unsqueeze(-1)
        log_probs_view1_corrected = log_softmax_spatial(scores_v1_corrected)
        B_pano, H, W, N_rot = log_probs_view1_corrected.shape
        view1_flat_spatial_corrected = log_probs_view1_corrected.view(B_pano, H * W, N_rot)
        yaw_log_heatmap = torch.logsumexp(view1_flat_spatial_corrected, dim=1)

        fused_scores = (1 - self.fusion_beta) * log_probs_pano + self.fusion_beta * yaw_log_heatmap.view(B_pano, 1, 1, N_rot)
        scores = log_softmax_spatial(fused_scores) # Re-normalize after fusion

        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)

        if "map_mask" in data:
            map_mask = data["map_mask"][::3] if is_panorama else data["map_mask"]
            scores.masked_fill_(~map_mask[..., None], -np.inf)
        if "yaw_prior" in data:
            yaw_prior = data["yaw_prior"][::3] if is_panorama else data["yaw_prior"]
            mask_yaw_prior(scores, yaw_prior, self.conf.num_rotations)
            
        log_probs = log_softmax_spatial(scores) # Re-normalize after masking
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev_individual": features_bev_individual,
            "valid_bev_individual": valid_bev_individual,
            "confidence_bev_individual": confidence_bev_individual,
            "features_bev": f_bev_final,
            "valid_bev": valid_bev_final,
            "confidence_bev": confidence_bev_final,
            "is_panorama": is_panorama,
        }

    def loss(self, pred, data):
        # Handle panorama batches - use ground truth from view1 of each panorama
        if pred.get("is_panorama", False):
            # For panorama batches, take ground truth from view1 (every 3rd element)
            xy_gt = data["uv"][::3]
            yaw_gt = data["roll_pitch_yaw"][::3, -1]
            mask = data.get("map_mask")
            if mask is not None:
                mask = mask[::3]
        else:
            # Standard single-view processing
            xy_gt = data["uv"]
            yaw_gt = data["roll_pitch_yaw"][..., -1]
            mask = data.get("map_mask")

        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=mask,
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }