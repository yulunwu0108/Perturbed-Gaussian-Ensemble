import sys
import torch
from tqdm import tqdm
from typing import List
from copy import deepcopy

sys.path.append("./")
from radiative_gaussian.gaussian.render_query import render
from radiative_gaussian.dataset import Scene
from radiative_gaussian.utils.loss_utils import ssim


class EnsembleSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.num_trials = int(getattr(args, "ensemble_num_trials", 10))
        self.low_density_ratio = float(getattr(args, "ensemble_low_density_ratio", 0.1))
        scale =  float(getattr(args, "ensemble_scale", 0.5))
        self.scale_min = 1.0 - scale
        self.scale_max = 1.0 + scale
        self.min_density = float(getattr(args, "ensemble_min_density", 1e-6))

    def _render_image_set(self, cameras, gaussians, pipe):
        images = []
        for cam in cameras:
            with torch.no_grad():
                render_pkg = render(cam, gaussians, pipe)
            images.append(render_pkg["render"].detach())
        return images

    def _camera_gaussian_contribution(self, gaussians, cam, pipe) -> torch.Tensor:
        render_pkg = render(cam, gaussians, pipe)
        pred_img = render_pkg["render"]
        grad_density = torch.autograd.grad(
            outputs=pred_img,
            inputs=gaussians._density,
            grad_outputs=torch.ones_like(pred_img),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        return grad_density.detach().reshape(-1).abs()

    def nbvs(self, gaussians, scene: Scene, num_views, pipe) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))
        candidate_cameras = scene.getCandidateCameras()
        if num_views <= 0 or len(candidate_views) == 0:
            return []

        num_candidates = len(candidate_cameras)
        base_candidate_images = self._render_image_set(candidate_cameras, gaussians, pipe)

        num_gaussians = gaussians._density.shape[0]
        if num_gaussians == 0:
            return candidate_views[: min(num_views, len(candidate_views))]

        density_full = gaussians.get_density.detach().clone()
        density_flat = density_full.reshape(-1)

        valid_gaussian_mask = torch.ones_like(density_flat, dtype=torch.bool)

        valid_density = density_flat[valid_gaussian_mask]
        valid_indices = torch.where(valid_gaussian_mask)[0]
        num_low = max(1, int(self.low_density_ratio * valid_density.numel()))
        num_low = min(num_low, valid_density.numel())
        low_density_idx = torch.topk(
            valid_density, k=num_low, largest=False, sorted=False
        ).indices
        low_density_idx = valid_indices[low_density_idx]

        ssim_trials = torch.zeros(
            (num_candidates, self.num_trials),
            device=density_flat.device,
            dtype=density_flat.dtype,
        )
        raw_density_backup = gaussians._density.detach().clone()

        for trial_id in tqdm(range(self.num_trials), desc="Ensemble perturbation trials"):
            with torch.no_grad():
                perturbed_density = density_flat.clone()
                scale = torch.empty(num_low, device=density_flat.device).uniform_(
                    self.scale_min, self.scale_max
                )
                perturbed_density[low_density_idx] *= scale
                perturbed_density = torch.clamp_min(perturbed_density, self.min_density)
                gaussians._density.copy_(
                    gaussians.density_inverse_activation(
                        perturbed_density.view_as(density_full)
                    )
                )

            perturbed_candidate_images = self._render_image_set(
                candidate_cameras, gaussians, pipe
            )
            for cam_idx, (base_img, perturbed_img) in enumerate(
                zip(base_candidate_images, perturbed_candidate_images)
            ):
                ssim_trials[cam_idx, trial_id] = ssim(base_img, perturbed_img).detach()

        with torch.no_grad():
            gaussians._density.copy_(raw_density_backup)

        score_var = torch.var(ssim_trials, dim=1, unbiased=False)
        best_idx = torch.argmax(score_var).item()

        return [candidate_views[best_idx]]

    def forward(self, x):
        return x
