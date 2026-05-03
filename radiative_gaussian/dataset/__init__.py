import os
import sys
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from radiative_gaussian.gaussian import GaussianModel
from radiative_gaussian.arguments import ModelParams
from radiative_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from radiative_gaussian.utils.camera_utils import cameraList_from_camInfos
from radiative_gaussian.utils.general_utils import t2a


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        num_views = len(scene_info.train_cameras)
        self.all_train_set = set(range(num_views))
        self.train_idxs = list(range(num_views))

        if shuffle:
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

        self.candidate_views_filter = None

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        filted_train_cameras = [self.train_cameras[i] for i in self.train_idxs]
        return filted_train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def get_candidate_set(self):
        # Get candidate set
        # Ensure resutls are always the same
        candidate_set = sorted(list(self.all_train_set - set(self.train_idxs)))
        if self.candidate_views_filter is not None:
            candidate_set = list(filter(self.candidate_views_filter, candidate_set))
        return candidate_set

    def getCandidateCameras(self):
        candidate_set = list(self.get_candidate_set())
        filted_train_cameras = [self.train_cameras[i] for i in candidate_set]
        return filted_train_cameras
