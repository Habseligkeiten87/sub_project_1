import numpy as np
import open3d as o3d
import os
import struct
import os.path as osp
import glob
import collections
from collections import defaultdict
from pprint import pprint
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm_base.vggsfm.runners.runner import VGGSfMRunner
from vggsfm_base.vggsfm.datasets.demo_loader import DemoLoader
from vggsfm_base.vggsfm.utils.utils import seed_all_random_engines

from utils.thread_utils import run_on_thread
from utils.read_write_model import colmap_to_open3d, read_cameras_binary, read_images_binary, read_points3D_binary


VOCAB_PATH = 'modules/colmap/vocab_tree_flickr100K_words32K.bin'

IMAGE_DIR = None

# NOTE: change the path here
@hydra.main(config_path="/home/xianhang.cheng/code/sub_project_1/assignment1/vggsfm_base/cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    global IMAGE_DIR
    print(IMAGE_DIR, "*****************************************\n")
    test_dataset = DemoLoader(
        SCENE_DIR=IMAGE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )

    sequence_list = test_dataset.sequence_list

    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    output_dir = batch[
        "scene_dir"
    ]  # which is also cfg.SCENE_DIR for DemoLoader

    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = (
        batch["crop_params"] if batch["crop_params"] is not None else None
    )

    # Cache the original images for visualization, so that we don't need to re-load many times
    original_images = batch["original_images"]

    # Run VGGSfM
    # Both visualization and output writing are performed inside VGGSfMRunner
    predictions = vggsfm_runner.run(
        images,
        masks=masks,
        original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir,
    )

    print("Demo Finished Successfully")

    return True




class ColmapAPI:
    def __init__(
        self,
        gpu_index,
        camera_model,
        matcher,
    ):
        self._data_path = None
        self._pcd = None
        self._thread = None
        self._active_camera_name = None
        self._cameras = dict()
        self._vis = None

        self._gpu_index = gpu_index
        self._camera_model = camera_model
        self._matcher = matcher
        if self._matcher not in ['exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher']:
            raise ValueError(f'Only support exhaustive_matcher and vocab_tree_matcher, got {self._matcher}')

    @property
    def data_path(self):
        if self._data_path is None:
            raise ValueError(f'Data path was not set')
        return self._data_path

    @data_path.setter
    def data_path(self, new_data_path):
        self._data_path = new_data_path

    @property
    def image_dir(self):
        return osp.join(self.data_path, 'images')

    @property
    def database_path(self):
        return osp.join(self.data_path, 'colmap/database.db')

    @property
    def sparse_dir(self):
        return osp.join(self.data_path, 'colmap/sparse')

    @property
    def num_cameras(self):
        return len(self._cameras)

    @property
    def camera_names(self):
        return list(self._cameras.keys())

    @property
    def pcd(self):
        if self._pcd is None:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._pcd

    @property
    def activate_camera_name(self):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        return self._active_camera_name

    @activate_camera_name.setter
    def activate_camera_name(self, new_value):
        if len(self._cameras) == 0:
            raise ValueError(f'COLMAP has not estimated the camera yet')
        self._active_camera_name = new_value

    @property
    def camera_model(self):
        return self._camera_model

    @camera_model.setter
    def camera_model(self, new_value):
        self._camera_model = new_value 

    @property
    def matcher(self):
        return self._matcher

    @matcher.setter
    def matcher(self, new_value):
        self._matcher = new_value

    def check_colmap_folder_valid(self):
        database_path = self.database_path
        image_dir = self.image_dir
        sparse_dir = self.sparse_dir

        print('Database file:', database_path)
        print('Image path:', image_dir)
        print('Bundle adjustment path:', sparse_dir)

        is_valid = \
            osp.isfile(database_path) and \
            osp.isdir(image_dir) and \
            osp.isdir(sparse_dir)

        return is_valid

    @run_on_thread
    def _estimate_cameras(self, recompute):
        print("run into _estimate_cameras!!!!!!!!!!!!!!!!!!!!\n")
        ''' Assignment 1

        In this assignment, you need to compute two things:
            pcd: A colored point cloud represented using open3d.geometry.PointCloud
            cameras: A dictionary of the following format:
                {
                    camera_name_01 [str]: {
                        'extrinsics': [rotation [Matrix 3x3], translation [Vector 3]]
                        'intrinsics': {
                            'width': int
                            'height': int
                            'fx': float
                            'fy': float
                            'cx': float
                            'cy': float
                        }
                    }
                    ...
                }

            You can check the extract_camera_parameters method to understand how the cameras are used.
        '''
        global IMAGE_DIR
        IMAGE_DIR = self.data_path
        with torch.no_grad():
            demo_fn()
        images_file = IMAGE_DIR + '/sparse/images.bin'
        points3D_file = IMAGE_DIR + '/sparse/points3D.bin'
        cameras_file = IMAGE_DIR + '/sparse/cameras.bin'

        cameras = read_cameras_binary(cameras_file)
        images = read_images_binary(images_file)
        points3D = read_points3D_binary(points3D_file)
        # print(cameras)
        #print(images[0])
        #pprint(points3D)

        ## Insert your code below
        if recompute:
            print("run into recompute!!!!!!!!!!!!!!!!!!!!!!!\n")
            # Compute the result once and cache it in self.data_path. This will save a lot of time on the next run
            # If you use COLMAP, save the database and bundle adjustment data in self.database_dir and
            # self.sparse_dir, respectively.
            pass

        # You can load the cached data here before adding points and cameras

        # Add points
        pcd = o3d.geometry.PointCloud()

        # Add cameras
        colmap_cameras = {}

        pcd, colmap_cameras = colmap_to_open3d(cameras,images,points3D)
        # print("*************************************\n\n\n")
        # pprint(pcd)
        ####### End of your code #####################

        self._pcd = pcd
        self._cameras = colmap_cameras
        self.activate_camera_name = self.camera_names[0]
        print("run out of _estimate_cameras!!!!!!!!!!!!!!!!!!!!\n")

    @staticmethod
    def _list_images_in_folder(directory):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}
        files = sorted(glob.glob(osp.join(directory, '*')))
        print("there are {} images in this folder\n".format(len(files)) )
        files = list(filter(lambda x: osp.splitext(x)[1].lower() in image_extensions, files)) 
        return files

    def estimate_done(self):
        return not self._thread.is_alive()

    def estimate_cameras(self, recompute=False):
        self._thread = self._estimate_cameras(recompute)

    def extract_camera_parameters(self, camera_name):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self._cameras[camera_name]['intrinsic']['width'],
            self._cameras[camera_name]['intrinsic']['height'],
            self._cameras[camera_name]['intrinsic']['fx'],
            self._cameras[camera_name]['intrinsic']['fy'],
            self._cameras[camera_name]['intrinsic']['cx'],
            self._cameras[camera_name]['intrinsic']['cy'],
        )

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = self._cameras[camera_name]['extrinsic'][0]
        extrinsics[:3, 3] = self._cameras[camera_name]['extrinsic'][1]
        extrinsics = extrinsics

        return intrinsics, extrinsics
