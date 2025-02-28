import unittest
import mujoco
import os
from multiverse_parser import boxify, voxelize, MjcfBoxify, UrdfBoxify

RESOURCES_PATH = "../resources/input"
OUTPUT_PATH = "../resources/output/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

class TestMjcfBoxify(unittest.TestCase):
    xml_path = f"{RESOURCES_PATH}/mujoco_menagerie/franka_fr3/fr3.xml"

    def test_import_scene(self) -> MjcfBoxify:
        mjcf_boxify = MjcfBoxify(self.xml_path)
        self.assertIsNotNone(mjcf_boxify.model)
        return mjcf_boxify

    def test_boxify_first_mesh(self):
        mjcf_boxify = self.test_import_scene()
        mesh_name = None
        for geom in mjcf_boxify.spec.geoms:
            if (geom.conaffinity != 0 or geom.contype != 0) and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_name = geom.meshname
                break
        self.assertIsNotNone(mesh_name)

        output_mesh_file = os.path.join(OUTPUT_PATH, f"{mesh_name}.obj")
        mjcf_boxify.boxify_mesh(mesh_name, output_mesh_file, threshold=0.75)

    def test_boxify_all_meshes(self):
        mjcf_boxify = self.test_import_scene()
        for geom in mjcf_boxify.spec.geoms:
            if (geom.conaffinity != 0 or geom.contype != 0) and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_name = geom.meshname
                output_mesh_file = os.path.join(OUTPUT_PATH, f"{mesh_name}.obj")
                mjcf_boxify.boxify_mesh(mesh_name, output_mesh_file, threshold=0.75)
                mjcf_boxify.get_cubes(output_mesh_file)

    def test_boxify_all_geoms(self):
        mjcf_boxify = self.test_import_scene()
        mjcf_boxify.boxify_all_meshes(threshold=0.5, visible=True)
        mjcf_boxify.remove_all_meshes()
        mjcf_boxify.save_as(os.path.join(OUTPUT_PATH, "fr3_boxified.xml"))

class TestBoxify(unittest.TestCase):
    def test_mesh_boxify(self):
        input_file_path = f"{RESOURCES_PATH}/meshes/IAIDrawerW60H53.stl"
        output_file_path = f"{OUTPUT_PATH}/IAIDrawerW60H53_boxified.obj"
        boxify(input_file_path, output_file_path, threshold=0.1, seed=3)

    def test_mjcf_boxify(self):
        file_path = f"{RESOURCES_PATH}/mujoco_menagerie/leap_hand/left_hand.xml"
        mjcf_boxify = MjcfBoxify(file_path)
        mjcf_boxify.boxify_all_meshes(threshold=0.5, visible=True)
        mjcf_boxify.remove_all_meshes()
        file_name = os.path.basename(file_path)
        file_name = file_name.split(".")[0]
        mjcf_boxify.save_as(os.path.join(OUTPUT_PATH, f"{file_name}_boxified.xml"))

    def test_urdf_boxify(self):
        file_path = f"{RESOURCES_PATH}/furniture/cabinet.urdf"
        urdf_boxify = UrdfBoxify(file_path)
        urdf_boxify.boxify_all_meshes(threshold=0.2, from_visual=False)
        urdf_boxify.remove_all_meshes()
        file_name = os.path.basename(file_path)
        file_name = file_name.split(".")[0]
        urdf_boxify.save_as(os.path.join(OUTPUT_PATH, f"{file_name}_boxified.urdf"))

class TestVoxelize(unittest.TestCase):
    def test_mesh_voxelize_stl(self):
        input_file_path = f"{RESOURCES_PATH}/meshes/IAIDrawerW60H53.stl"
        output_file_path = f"{OUTPUT_PATH}/IAIDrawerW60H53_voxelized.stl"
        voxelize(input_file_path, output_file_path, 5)
        self.assertTrue(os.path.exists(output_file_path))

    def test_mesh_voxelize_obj(self):
        input_file_path = f"{RESOURCES_PATH}/furniture/obj/IAICabinet1W60.obj"
        output_file_path = f"{OUTPUT_PATH}/IAICabinet1W60_voxelized.stl"
        voxelize(input_file_path, output_file_path,7)
        self.assertTrue(os.path.exists(output_file_path))