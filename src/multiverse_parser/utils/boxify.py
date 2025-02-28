#!/usr/bin/env python3

from typing import List, Tuple

from pxr.Tf import RaiseCodingError

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    exit(1)

import numpy
import os
import coacd
import mujoco
from urdf_parser_py import urdf
import xml.etree.ElementTree as ET
import bpy
from scipy.spatial.transform import Rotation
import numpy as np


def boxify(input_file: str,
           output_file: str,
           quiet: bool = False,
           threshold: float = 0.75,
           max_convex_hull: int = 20,
           preprocess_mode: str = "auto",
           prep_resolution: int = 50,
           resolution: int = 2000,
           mcts_node: int = 20,
           mcts_iteration: int = 150,
           mcts_max_depth: int = 3,
           pca: bool = False,
           no_merge: bool = True,
           decimate: bool = True,
           max_ch_vertex: int = 256,
           extrude: bool = False,
           extrude_margin: float = 0.01,
           apx_mode: str = "box",
           seed: int = 0):
    """
    Boxify a mesh file

    :param input_file: input mesh file (OBJ or STL)
    :param output_file: output mesh file (OBJ)
    :param quiet: suppress log messages
    :param threshold: concavity threshold for terminating the decomposition (0.01~1)
    :param max_convex_hull: maximum number of convex hulls to generate
    :param preprocess_mode: choose manifold preprocessing mode ('auto': automatically check input mesh manifoldness; 'on': force turn on the pre-processing; 'off': force turn off the pre-processing), default = 'auto'.
    :param prep_resolution: resolution for manifold preprocess (20~100), default = 50.
    :param resolution: sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000.
    :param mcts_node: max number of child nodes in MCTS (10~40), default = 20.
    :param mcts_iteration: number of search iterations in MCTS (60~2000), default = 100.
    :param mcts_max_depth: max search depth in MCTS (2~7), default = 3.
    :param pca: flag to enable PCA pre-processing, default = false.
    :param no_merge: flag to disable merge postprocessing, default = true.
    :param decimate: enable max vertex constraint per convex hull, default = true.
    :param max_ch_vertex: max vertex value for each convex hull, only when decimate is enabled, default = 256.
    :param extrude: extrude neighboring convex hulls along the overlapping faces (other faces unchanged), default = false.
    :param extrude_margin: extrude margin, only when extrude is enabled, default = 0.01.
    :param apx_mode: approximation shape type ("ch" for convex hulls, "box" for cubes), default = "box".
    :param seed: random seed used for sampling, default = 0.
    """
    if not os.path.isfile(input_file):
        print(input_file, "is not a file")
        exit(1)

    if quiet:
        coacd.set_log_level("error")

    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=prep_resolution,
        resolution=resolution,
        mcts_nodes=mcts_node,
        mcts_iterations=mcts_iteration,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=not no_merge,
        decimate=decimate,
        max_ch_vertex=max_ch_vertex,
        extrude=extrude,
        extrude_margin=extrude_margin,
        apx_mode=apx_mode,
        seed=seed,
    )
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    numpy.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (numpy.random.rand(3) * 255).astype(numpy.uint8)
        scene.add_geometry(p)
    scene.export(output_file)


def voxelize(input_file: str,
             output_file: str,
             octree_depth: int) -> None:
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for from_obj in bpy.data.objects:
        bpy.data.objects.remove(from_obj)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    if input_file.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=input_file, up_axis='Z', forward_axis='Y')
        obj = bpy.context.object
        if obj and obj.type == 'MESH':
            apply_remesh = obj.modifiers.new(name="Remesh", type="REMESH")
            apply_remesh.mode = "BLOCKS"
            apply_remesh.octree_depth = octree_depth
            apply_remesh.scale = 1.0
            bpy.ops.object.modifier_apply(modifier=apply_remesh.name)
            bpy.ops.wm.stl_export(filepath=output_file)
    elif input_file.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=input_file, up_axis='Z', forward_axis='Y')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        obj = bpy.context.object
        if obj and obj.type == 'MESH':
            apply_remesh = obj.modifiers.new(name="Remesh", type="REMESH")
            apply_remesh.mode = "BLOCKS"
            apply_remesh.octree_depth = octree_depth
            apply_remesh.scale = 1.0
            bpy.ops.object.modifier_apply(modifier=apply_remesh.name)
            bpy.ops.wm.stl_export(filepath=output_file)


class Boxify:
    def __init__(self, file_path: str):
        """
        Initialize the Boxify object

        :param file_path: path to the file to be boxified
        """
        self.file_path = file_path

    @staticmethod
    def get_cubes(file_path: str) -> List[Tuple[List[float], List[float]]]:
        for armature in bpy.data.armatures:
            bpy.data.armatures.remove(armature)
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        for from_obj in bpy.data.objects:
            bpy.data.objects.remove(from_obj)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
        for camera in bpy.data.cameras:
            bpy.data.cameras.remove(camera)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        for image in bpy.data.images:
            bpy.data.images.remove(image)

        file_ext = os.path.splitext(file_path)[1]
        if file_ext == ".usd" or file_ext == ".usda":
            bpy.ops.wm.usd_import(filepath=file_path, scale=1.0)
        elif file_ext == ".dae":
            bpy.ops.wm.collada_import(filepath=file_path)
        elif file_ext == ".obj":
            bpy.ops.wm.obj_import(filepath=file_path, up_axis='Z', forward_axis='Y')

        cubes = []

        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mesh = obj.data

                world_vertices = [obj.matrix_world @ v.co for v in mesh.vertices]

                min_corner = [min(v[i] for v in world_vertices) for i in range(3)]
                max_corner = [max(v[i] for v in world_vertices) for i in range(3)]

                origin = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
                size = [max_corner[i] - min_corner[i] for i in range(3)]
                cubes.append((origin, size))

        return cubes

    @staticmethod
    def get_cubes_from_voxelized(file_path: str) -> List[Tuple[List[float], List[float]]]:
        """
        Generate cubes from voxelized mesh

        :param file_path: File path to the voxelized mesh
        :return: List of cubes (origin, size)
        """
        for armature in bpy.data.armatures:
            bpy.data.armatures.remove(armature)
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        for from_obj in bpy.data.objects:
            bpy.data.objects.remove(from_obj)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
        for camera in bpy.data.cameras:
            bpy.data.cameras.remove(camera)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        for image in bpy.data.images:
            bpy.data.images.remove(image)
        bpy.ops.wm.stl_import(filepath=file_path, up_axis='Z', forward_axis='Y')
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        voxel_keys = []
        voxel_dict = {}
        for i, vert in enumerate(obj.data.vertices):
            voxel_key = [float(vert.co.x), float(vert.co.y), float(vert.co.z)]
            voxel_keys.append(voxel_key)
        size_list = len(voxel_keys)
        if size_list % 2 == 0:
            v1 = voxel_keys[size_list // 2]
            v2 = voxel_keys[(size_list // 2) - 1]
        else:
            size_list = size_list - 1
            v1 = voxel_keys[size_list // 2]
            v2 = voxel_keys[(size_list // 2) - 1]
        x = v1[0] - v2[0]
        y = v1[1] - v2[1]
        z = v1[2] - v2[2]
        longitude = None
        for value in [x, y, z]:
            if value != 0:
                longitude = value
                break
        cubes = []
        size = [longitude, longitude, longitude]
        for i in voxel_keys:
            location = i
            cubes.append((location, size))
        return cubes

    def remove_all_meshes(self):
        """
        Remove all meshes from the file
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def save_as(self, file_path: str):
        """
        Save the file to a new location
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    @property
    def file_dir(self) -> str:
        return os.path.dirname(self.file_path)


class MjcfBoxify(Boxify):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.spec: mujoco.MjSpec = mujoco.MjSpec.from_file(self.file_path)
        self.model: mujoco.MjModel = self.spec.compile()

    def boxify_mesh(self, mesh_name: str, output_path: str, threshold: float = 0.75):
        """
        Boxify a mesh in the MJCF file

        :param mesh_name: name of the mesh to be boxified
        :param output_path: path to save the cube
        :param threshold: concavity threshold for terminating the decomposition (0.01~1)
        """
        mesh_id = self.model.mesh(mesh_name).id
        mesh_spec = self.spec.meshes[mesh_id]
        mesh_dir = self.spec.meshdir
        if not os.path.isabs(mesh_dir):
            mesh_dir = os.path.join(self.file_dir, self.spec.meshdir)
        mesh_path = mesh_spec.file
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(mesh_dir, mesh_path)
        boxify(mesh_path, output_path, threshold=threshold)

    def boxify_all_meshes(self, threshold: float = 0.75, visible: bool = True):
        """
        Boxify all meshes in the MJCF file

        :param threshold: concavity threshold for terminating the decomposition (0.01~1)
        :param visible: whether the cubes should be visible
        """
        for body in self.spec.bodies:
            body: mujoco.MjsBody
            for geom in body.geoms:
                if (geom.conaffinity != 0 or geom.contype != 0) and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_name = geom.meshname
                    output_file = os.path.join(self.file_dir, f"{mesh_name}.obj")
                    self.boxify_mesh(mesh_name, output_file, threshold=threshold)
                    for i, (cube_origin, cube_size) in enumerate(self.get_cubes(output_file)):
                        geom_name = f"{body.name}_cube_{i}"
                        cube_size = [s * 0.5 for s in cube_size]
                        body.add_geom(
                            name=geom_name,
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=cube_size,
                            pos=cube_origin,
                            rgba=[0.9, 0.9, 0.9, 1.0],
                            group=geom.group if not visible else 0,
                            conaffinity=geom.conaffinity,
                            contype=geom.contype,
                        )
                    geom.delete()
                    os.remove(output_file)

    def remove_all_meshes(self):
        for body in self.spec.bodies:
            body: mujoco.MjsBody
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    geom.delete()
        for mesh in self.spec.meshes:
            mesh.delete()

    def save_as(self, file_path: str):
        self.spec.compile()
        xml_string = self.spec.to_xml()
        with open(file_path, "w") as f:
            f.write(xml_string)


class UrdfBoxify(Boxify):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.robot: urdf.Robot = urdf.URDF.from_xml_file(file_path)
        self.tree = ET.parse(self.file_path)

    def boxify_mesh(self, mesh_path: str, output_path: str, threshold: float = 0.75):
        """
        Boxify a mesh in the URDF file

        :param mesh_path: path to the mesh to be boxified
        :param output_path: path to save the cube
        :param threshold: concavity threshold for terminating the decomposition (0.01~1)
        """
        if mesh_path.startswith("package://"):
            raise NotImplementedError("package:// is not supported yet")
        elif mesh_path.startswith("file://"):
            mesh_path = mesh_path[7:]
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(self.file_dir, mesh_path)
        boxify(mesh_path, output_path, threshold=threshold)

    def boxify_all_meshes(self, threshold: float = 0.75, from_visual: bool = True):
        """
        Boxify all meshes in the URDF file

        :param threshold: concavity threshold for terminating the decomposition (0.01~1)
        :param from_visual: whether to boxify the visual or collision meshes
        """
        for link in self.robot.links:
            link: urdf.Link
            geoms = link.visuals if from_visual else link.collisions
            for geom in geoms:
                if type(geom.geometry) is urdf.Mesh:
                    mesh_file_path = geom.geometry.filename
                    mesh_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
                    output_file = os.path.join(self.file_dir, f"{mesh_name}.obj")
                    self.boxify_mesh(mesh_file_path, output_file, threshold=threshold)
                    origin = urdf.Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]) \
                        if not hasattr(geom, "origin") or geom.origin is None else geom.origin
                    for i, (cube_origin, cube_size) in enumerate(self.get_cubes(output_file)):
                        geom_name = f"{link.name}_cube_{i}"
                        origin_rotation = Rotation.from_euler("xyz", origin.rpy)
                        cube_xyz = origin_rotation.apply(cube_origin) + origin.xyz
                        cube_origin = urdf.Pose(xyz=cube_xyz, rpy=origin.rpy)
                        geometry = urdf.Box(size=cube_size)
                        collision = urdf.Collision(
                            geometry=geometry,
                            origin=cube_origin,
                            name=geom_name)
                        link.collision = collision
                    os.remove(output_file)

    def remove_all_meshes(self):
        xml_string = self.robot.to_xml_string()
        self.tree = ET.ElementTree(ET.fromstring(xml_string))
        root = self.tree.getroot()

        for link in root.findall(".//link"):
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None and geometry.find("mesh") is not None:
                    link.remove(visual)
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is not None and geometry.find("mesh") is not None:
                    link.remove(collision)

    def save_as(self, file_path: str):
        self.tree.write(file_path)
