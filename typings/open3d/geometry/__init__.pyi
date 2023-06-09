from typing import Any, Callable, ClassVar, List

from typing import overload
import numpy
import open3d.cpu.pybind.utility
import open3d.geometry
All: FilterScope
Average: SimplificationContraction
Color: FilterScope
Gaussian3: ImageFilterType
Gaussian5: ImageFilterType
Gaussian7: ImageFilterType
Normal: FilterScope
Quadric: SimplificationContraction
Smoothed: DeformAsRigidAsPossibleEnergy
Sobel3dx: ImageFilterType
Sobel3dy: ImageFilterType
Spokes: DeformAsRigidAsPossibleEnergy
Vertex: FilterScope

class AxisAlignedBoundingBox(Geometry3D):
    color: numpy.ndarray[numpy.float64[3,1]]
    max_bound: numpy.ndarray[numpy.float64[3,1]]
    min_bound: numpy.ndarray[numpy.float64[3,1]]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: AxisAlignedBoundingBox) -> None: ...
    @overload
    def __init__(self, min_bound: numpy.ndarray[numpy.float64[3,1]], max_bound: numpy.ndarray[numpy.float64[3,1]]) -> None: ...
    def create_from_points(self, *args, **kwargs) -> Any: ...
    def get_box_points(self) -> Any: ...
    def get_extent(self) -> Any: ...
    def get_half_extent(self) -> Any: ...
    def get_max_extent(self) -> Any: ...
    def get_point_indices_within_bounding_box(self, points) -> Any: ...
    def get_print_info(self) -> Any: ...
    def volume(self) -> Any: ...
    def __copy__(self) -> AxisAlignedBoundingBox: ...
    def __deepcopy__(self, arg0: dict) -> AxisAlignedBoundingBox: ...
    def __iadd__(self, arg0: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox: ...

class DeformAsRigidAsPossibleEnergy:
    __members__: ClassVar[dict] = ...  # read-only
    Smoothed: ClassVar[DeformAsRigidAsPossibleEnergy] = ...
    Spokes: ClassVar[DeformAsRigidAsPossibleEnergy] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class FilterScope:
    __members__: ClassVar[dict] = ...  # read-only
    All: ClassVar[FilterScope] = ...
    Color: ClassVar[FilterScope] = ...
    Normal: ClassVar[FilterScope] = ...
    Vertex: ClassVar[FilterScope] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Geometry:
    class Type:
        __members__: ClassVar[dict] = ...  # read-only
        HalfEdgeTriangleMesh: ClassVar[Geometry.Type] = ...
        Image: ClassVar[Geometry.Type] = ...
        LineSet: ClassVar[Geometry.Type] = ...
        PointCloud: ClassVar[Geometry.Type] = ...
        RGBDImage: ClassVar[Geometry.Type] = ...
        TetraMesh: ClassVar[Geometry.Type] = ...
        TriangleMesh: ClassVar[Geometry.Type] = ...
        Unspecified: ClassVar[Geometry.Type] = ...
        VoxelGrid: ClassVar[Geometry.Type] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: int) -> None: ...
        def __eq__(self, other: object) -> bool: ...
        def __ge__(self, other: object) -> bool: ...
        def __getstate__(self) -> int: ...
        def __gt__(self, other: object) -> bool: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __int__(self) -> int: ...
        def __le__(self, other: object) -> bool: ...
        def __lt__(self, other: object) -> bool: ...
        def __ne__(self, other: object) -> bool: ...
        def __setstate__(self, state: int) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...
    HalfEdgeTriangleMesh: ClassVar[Geometry.Type] = ...
    Image: ClassVar[Geometry.Type] = ...
    LineSet: ClassVar[Geometry.Type] = ...
    PointCloud: ClassVar[Geometry.Type] = ...
    RGBDImage: ClassVar[Geometry.Type] = ...
    TetraMesh: ClassVar[Geometry.Type] = ...
    TriangleMesh: ClassVar[Geometry.Type] = ...
    Unspecified: ClassVar[Geometry.Type] = ...
    VoxelGrid: ClassVar[Geometry.Type] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Any: ...
    def dimension(self) -> Any: ...
    def get_geometry_type(self) -> Any: ...
    def is_empty(self) -> Any: ...

class Geometry2D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_max_bound(self) -> Any: ...
    def get_min_bound(self) -> Any: ...

class Geometry3D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_axis_aligned_bounding_box(self) -> Any: ...
    def get_center(self) -> Any: ...
    def get_max_bound(self) -> Any: ...
    def get_min_bound(self) -> Any: ...
    def get_minimal_oriented_bounding_box(self, *args, **kwargs) -> Any: ...
    def get_oriented_bounding_box(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_axis_angle(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_quaternion(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_xyz(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_xzy(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_yxz(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_yzx(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_zxy(self, *args, **kwargs) -> Any: ...
    def get_rotation_matrix_from_zyx(self, *args, **kwargs) -> Any: ...
    @overload
    def rotate(self, R) -> Any: ...
    @overload
    def rotate(self, R, center) -> Any: ...
    @overload
    def scale(self, scale, center) -> Any: ...
    @overload
    def scale(float) -> Any: ...
    @overload
    def scale(self, scale, center) -> Any: ...
    @overload
    def scale(float) -> Any: ...
    def transform(self, arg0) -> Any: ...
    def translate(self, translation, relative = ...) -> Any: ...

class HalfEdge:
    next: int
    triangle_index: int
    twin: int
    vertex_indices: numpy.ndarray[numpy.int32[2,1]]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: HalfEdge) -> None: ...
    def is_boundary(self) -> bool: ...
    def __copy__(self) -> HalfEdge: ...
    def __deepcopy__(self, arg0: dict) -> HalfEdge: ...

class HalfEdgeTriangleMesh(MeshBase):
    half_edges: List[HalfEdge]
    ordered_half_edge_from_vertex: List[open3d.cpu.pybind.utility.IntVector]
    triangle_normals: open3d.cpu.pybind.utility.Vector3dVector
    triangles: open3d.cpu.pybind.utility.Vector3iVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: HalfEdgeTriangleMesh) -> None: ...
    def boundary_half_edges_from_vertex(self, vertex_index) -> Any: ...
    def boundary_vertices_from_vertex(self) -> Any: ...
    def create_from_triangle_mesh(self, *args, **kwargs) -> Any: ...
    def get_boundaries(self) -> Any: ...
    def has_half_edges(self) -> Any: ...
    def __copy__(self) -> HalfEdgeTriangleMesh: ...
    def __deepcopy__(self, arg0: dict) -> HalfEdgeTriangleMesh: ...

class Image(Geometry2D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: Image) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    def create_pyramid(self, num_of_levels, with_gaussian_filter) -> Any: ...
    def filter(self, filter_type) -> Any: ...
    def filter_pyramid(self, *args, **kwargs) -> Any: ...
    def flip_horizontal(self) -> Image: ...
    def flip_vertical(self) -> Image: ...
    def __copy__(self) -> Image: ...
    def __deepcopy__(self, arg0: dict) -> Image: ...

class ImageFilterType:
    __members__: ClassVar[dict] = ...  # read-only
    Gaussian3: ClassVar[ImageFilterType] = ...
    Gaussian5: ClassVar[ImageFilterType] = ...
    Gaussian7: ClassVar[ImageFilterType] = ...
    Sobel3dx: ClassVar[ImageFilterType] = ...
    Sobel3dy: ClassVar[ImageFilterType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class KDTreeFlann:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, data: numpy.ndarray[numpy.float64[m,n]]) -> None: ...
    @overload
    def __init__(self, geometry: Geometry) -> None: ...
    @overload
    def __init__(self, feature) -> None: ...
    def search_hybrid_vector_3d(self, query, radius, max_nn) -> Any: ...
    def search_hybrid_vector_xd(self, query, radius, max_nn) -> Any: ...
    def search_knn_vector_3d(self, query, knn) -> Any: ...
    def search_knn_vector_xd(self, query, knn) -> Any: ...
    def search_radius_vector_3d(self, query, radius) -> Any: ...
    def search_radius_vector_xd(self, query, radius) -> Any: ...
    def search_vector_3d(self, query, search_param) -> Any: ...
    def search_vector_xd(self, query, search_param) -> Any: ...
    def set_feature(self, feature) -> Any: ...
    def set_geometry(self, geometry) -> Any: ...
    def set_matrix_data(self, data) -> Any: ...

class KDTreeSearchParam:
    class Type:
        __members__: ClassVar[dict] = ...  # read-only
        HybridSearch: ClassVar[KDTreeSearchParam.Type] = ...
        KNNSearch: ClassVar[KDTreeSearchParam.Type] = ...
        RadiusSearch: ClassVar[KDTreeSearchParam.Type] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: int) -> None: ...
        def __eq__(self, other: object) -> bool: ...
        def __ge__(self, other: object) -> bool: ...
        def __getstate__(self) -> int: ...
        def __gt__(self, other: object) -> bool: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        def __int__(self) -> int: ...
        def __le__(self, other: object) -> bool: ...
        def __lt__(self, other: object) -> bool: ...
        def __ne__(self, other: object) -> bool: ...
        def __setstate__(self, state: int) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...
    HybridSearch: ClassVar[KDTreeSearchParam.Type] = ...
    KNNSearch: ClassVar[KDTreeSearchParam.Type] = ...
    RadiusSearch: ClassVar[KDTreeSearchParam.Type] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def get_search_type(self) -> Any: ...

class KDTreeSearchParamHybrid(KDTreeSearchParam):
    max_nn: int
    radius: float
    def __init__(self, radius: float, max_nn: int) -> None: ...

class KDTreeSearchParamKNN(KDTreeSearchParam):
    knn: int
    def __init__(self, knn: int = ...) -> None: ...

class KDTreeSearchParamRadius(KDTreeSearchParam):
    radius: float
    def __init__(self, radius: float) -> None: ...

class LineSet(Geometry3D):
    colors: open3d.cpu.pybind.utility.Vector3dVector
    lines: open3d.cpu.pybind.utility.Vector2iVector
    points: open3d.cpu.pybind.utility.Vector3dVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: LineSet) -> None: ...
    @overload
    def __init__(self, points: open3d.cpu.pybind.utility.Vector3dVector, lines: open3d.cpu.pybind.utility.Vector2iVector) -> None: ...
    def create_camera_visualization(self, *args, **kwargs) -> Any: ...
    def create_from_axis_aligned_bounding_box(self, *args, **kwargs) -> Any: ...
    def create_from_oriented_bounding_box(self, *args, **kwargs) -> Any: ...
    def create_from_point_cloud_correspondences(self, *args, **kwargs) -> Any: ...
    def create_from_tetra_mesh(self, *args, **kwargs) -> Any: ...
    def create_from_triangle_mesh(self, *args, **kwargs) -> Any: ...
    def get_line_coordinate(self, line_index) -> Any: ...
    def has_colors(self) -> Any: ...
    def has_lines(self) -> Any: ...
    def has_points(self) -> Any: ...
    def paint_uniform_color(self, color) -> Any: ...
    def __add__(self, arg0: LineSet) -> LineSet: ...
    def __copy__(self) -> LineSet: ...
    def __deepcopy__(self, arg0: dict) -> LineSet: ...
    def __iadd__(self, arg0: LineSet) -> LineSet: ...

class MeshBase(Geometry3D):
    vertex_colors: open3d.cpu.pybind.utility.Vector3dVector
    vertex_normals: open3d.cpu.pybind.utility.Vector3dVector
    vertices: open3d.cpu.pybind.utility.Vector3dVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: MeshBase) -> None: ...
    def compute_convex_hull(self) -> Any: ...
    def has_vertex_colors(self) -> Any: ...
    def has_vertex_normals(self) -> Any: ...
    def has_vertices(self) -> Any: ...
    def normalize_normals(self) -> Any: ...
    def paint_uniform_color(self, color) -> Any: ...
    def __add__(self, arg0: MeshBase) -> MeshBase: ...
    def __copy__(self) -> MeshBase: ...
    def __deepcopy__(self, arg0: dict) -> MeshBase: ...
    def __iadd__(self, arg0: MeshBase) -> MeshBase: ...

class Octree(Geometry3D):
    max_depth: int
    origin: numpy.ndarray[numpy.float64[3,1]]
    root_node: OctreeNode
    size: float
    @overload
    def __init__(self) -> Any: ...
    @overload
    def __init__(self, arg0) -> Any: ...
    @overload
    def __init__(self, max_depth) -> Any: ...
    @overload
    def __init__(self, max_depth, origin, size) -> Any: ...
    def convert_from_point_cloud(self, point_cloud, size_expand = ...) -> Any: ...
    def create_from_voxel_grid(self) -> Any: ...
    def insert_point(self, point, f_init, f_update, fi_init = ..., fi_update = ...) -> Any: ...
    def is_point_in_bound(self, *args, **kwargs) -> Any: ...
    def locate_leaf_node(self, point) -> Any: ...
    def to_voxel_grid(self) -> Any: ...
    def traverse(self, f: Callable[[OctreeNode,OctreeNodeInfo],bool]) -> None: ...
    def __copy__(self) -> Octree: ...
    def __deepcopy__(self, arg0: dict) -> Octree: ...

class OctreeColorLeafNode(OctreeLeafNode):
    color: numpy.ndarray[numpy.float64[3,1]]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: OctreeColorLeafNode) -> None: ...
    def get_init_function(self, *args, **kwargs) -> Any: ...
    def get_update_function(self, *args, **kwargs) -> Any: ...
    def __copy__(self) -> OctreeColorLeafNode: ...
    def __deepcopy__(self, arg0: dict) -> OctreeColorLeafNode: ...

class OctreeInternalNode(OctreeNode):
    children: List[OctreeNode]
    @overload
    def __init__(self) -> Any: ...
    @overload
    def __init__(self, arg0) -> Any: ...
    def get_init_function(self, *args, **kwargs) -> Any: ...
    def get_update_function(self, *args, **kwargs) -> Any: ...
    def __copy__(self) -> OctreeInternalNode: ...
    def __deepcopy__(self, arg0: dict) -> OctreeInternalNode: ...

class OctreeInternalPointNode(OctreeInternalNode):
    children: List[OctreeNode]
    indices: List[int]
    @overload
    def __init__(self) -> Any: ...
    @overload
    def __init__(self, arg0) -> Any: ...
    def get_init_function(self, *args, **kwargs) -> Any: ...
    def get_update_function(self, *args, **kwargs) -> Any: ...
    def __copy__(self) -> OctreeInternalPointNode: ...
    def __deepcopy__(self, arg0: dict) -> OctreeInternalPointNode: ...

class OctreeLeafNode(OctreeNode):
    __hash__: ClassVar[None] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def clone(self) -> OctreeLeafNode: ...
    def __eq__(self, other: OctreeLeafNode) -> bool: ...

class OctreeNode:
    def __init__(self, *args, **kwargs) -> None: ...

class OctreeNodeInfo:
    child_index: int
    depth: int
    origin: numpy.ndarray[numpy.float64[3,1]]
    size: float
    def __init__(self, origin, size, depth, child_index) -> Any: ...

class OctreePointColorLeafNode(OctreeLeafNode):
    color: numpy.ndarray[numpy.float64[3,1]]
    indices: List[int]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: OctreePointColorLeafNode) -> None: ...
    def get_init_function(self, *args, **kwargs) -> Any: ...
    def get_update_function(self, *args, **kwargs) -> Any: ...
    def __copy__(self) -> OctreePointColorLeafNode: ...
    def __deepcopy__(self, arg0: dict) -> OctreePointColorLeafNode: ...

class OrientedBoundingBox(Geometry3D):
    R: numpy.ndarray[numpy.float64[3,3]]
    center: numpy.ndarray[numpy.float64[3,1]]
    color: numpy.ndarray[numpy.float64[3,1]]
    extent: numpy.ndarray[numpy.float64[3,1]]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: OrientedBoundingBox) -> None: ...
    @overload
    def __init__(self, center: numpy.ndarray[numpy.float64[3,1]], R: numpy.ndarray[numpy.float64[3,3]], extent: numpy.ndarray[numpy.float64[3,1]]) -> None: ...
    def create_from_axis_aligned_bounding_box(self, *args, **kwargs) -> Any: ...
    def create_from_points(self, *args, **kwargs) -> Any: ...
    def get_box_points(self) -> Any: ...
    def get_point_indices_within_bounding_box(self, points) -> Any: ...
    def volume(self) -> Any: ...
    def __copy__(self) -> OrientedBoundingBox: ...
    def __deepcopy__(self, arg0: dict) -> OrientedBoundingBox: ...

class PointCloud(Geometry3D):
    colors: open3d.cpu.pybind.utility.Vector3dVector
    covariances: open3d.cpu.pybind.utility.Matrix3dVector
    normals: open3d.cpu.pybind.utility.Vector3dVector
    points: open3d.cpu.pybind.utility.Vector3dVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: PointCloud) -> None: ...
    @overload
    def __init__(self, points: open3d.cpu.pybind.utility.Vector3dVector) -> None: ...
    def cluster_dbscan(self, eps, min_points, print_progress = ...) -> Any: ...
    def compute_convex_hull(self, *args, **kwargs) -> Any: ...
    def compute_mahalanobis_distance(self) -> Any: ...
    def compute_mean_and_covariance(self) -> Any: ...
    def compute_nearest_neighbor_distance(self) -> Any: ...
    def compute_point_cloud_distance(self, target) -> Any: ...
    def create_from_depth_image(self, *args, **kwargs) -> Any: ...
    def create_from_rgbd_image(self, *args, **kwargs) -> Any: ...
    @overload
    def crop(self, bounding_box) -> Any: ...
    @overload
    def crop(self, bounding_box) -> Any: ...
    def detect_planar_patches(self, normal_variance_threshold_deg = ..., coplanarity_deg = ..., outlier_ratio = ..., min_plane_edge_length = ..., min_num_points = ..., search_param = ...) -> Any: ...
    def estimate_covariances(self, search_param = ...) -> Any: ...
    def estimate_normals(self, search_param = ..., fast_normal_computation = ...) -> Any: ...
    def estimate_point_covariances(self, *args, **kwargs) -> Any: ...
    def farthest_point_down_sample(self, num_samples: int) -> PointCloud: ...
    def has_colors(self) -> Any: ...
    def has_covariances(self) -> bool: ...
    def has_normals(self) -> Any: ...
    def has_points(self) -> Any: ...
    def hidden_point_removal(self, camera_location, radius) -> Any: ...
    def normalize_normals(self) -> Any: ...
    def orient_normals_consistent_tangent_plane(self, k) -> Any: ...
    def orient_normals_to_align_with_direction(self, orientation_reference = ...) -> Any: ...
    def orient_normals_towards_camera_location(self, camera_location = ...) -> Any: ...
    def paint_uniform_color(self, color) -> Any: ...
    def random_down_sample(self, sampling_ratio) -> Any: ...
    def remove_duplicated_points(self) -> Any: ...
    def remove_non_finite_points(self, remove_nan = ..., remove_infinite = ...) -> Any: ...
    def remove_radius_outlier(self, nb_points, radius, print_progress = ...) -> Any: ...
    def remove_statistical_outlier(self, nb_neighbors, std_ratio, print_progress = ...) -> Any: ...
    def segment_plane(self, distance_threshold, ransac_n, num_iterations, probability = ...) -> Any: ...
    def select_by_index(self, indices, invert = ...) -> Any: ...
    def uniform_down_sample(self, every_k_points) -> Any: ...
    def voxel_down_sample(self, voxel_size) -> Any: ...
    def voxel_down_sample_and_trace(self, voxel_size, min_bound, max_bound, approximate_class = ...) -> Any: ...
    def __add__(self, arg0: PointCloud) -> PointCloud: ...
    def __copy__(self) -> PointCloud: ...
    def __deepcopy__(self, arg0: dict) -> PointCloud: ...
    def __iadd__(self, arg0: PointCloud) -> PointCloud: ...

class RGBDImage(Geometry2D):
    color: open3d.geometry.Image
    depth: open3d.geometry.Image
    def __init__(self) -> None: ...
    def create_from_color_and_depth(self, *args, **kwargs) -> Any: ...
    def create_from_nyu_format(self, *args, **kwargs) -> Any: ...
    def create_from_redwood_format(self, *args, **kwargs) -> Any: ...
    def create_from_sun_format(self, *args, **kwargs) -> Any: ...
    def create_from_tum_format(self, *args, **kwargs) -> Any: ...

class SimplificationContraction:
    __members__: ClassVar[dict] = ...  # read-only
    Average: ClassVar[SimplificationContraction] = ...
    Quadric: ClassVar[SimplificationContraction] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class TetraMesh(MeshBase):
    tetras: open3d.cpu.pybind.utility.Vector4iVector
    vertices: open3d.cpu.pybind.utility.Vector3dVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: TetraMesh) -> None: ...
    @overload
    def __init__(self, vertices: open3d.cpu.pybind.utility.Vector3dVector, tetras: open3d.cpu.pybind.utility.Vector4iVector) -> None: ...
    def create_from_point_cloud(self, *args, **kwargs) -> Any: ...
    def extract_triangle_mesh(self, values, level) -> Any: ...
    def has_tetras(self) -> Any: ...
    def has_vertices(self) -> Any: ...
    def remove_degenerate_tetras(self) -> Any: ...
    def remove_duplicated_tetras(self) -> Any: ...
    def remove_duplicated_vertices(self) -> Any: ...
    def remove_unreferenced_vertices(self) -> Any: ...
    def __add__(self, arg0: TetraMesh) -> TetraMesh: ...
    def __copy__(self) -> TetraMesh: ...
    def __deepcopy__(self, arg0: dict) -> TetraMesh: ...
    def __iadd__(self, arg0: TetraMesh) -> TetraMesh: ...

class TriangleMesh(MeshBase):
    adjacency_list: List of Sets
    textures: open3d.geometry.Image
    triangle_material_ids: open3d.cpu.pybind.utility.IntVector
    triangle_normals: open3d.cpu.pybind.utility.Vector3dVector
    triangle_uvs: open3d.cpu.pybind.utility.Vector2dVector
    triangles: open3d.cpu.pybind.utility.Vector3iVector
    vertex_colors: open3d.cpu.pybind.utility.Vector3dVector
    vertex_normals: open3d.cpu.pybind.utility.Vector3dVector
    vertices: open3d.cpu.pybind.utility.Vector3dVector
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: TriangleMesh) -> None: ...
    @overload
    def __init__(self, vertices: open3d.cpu.pybind.utility.Vector3dVector, triangles: open3d.cpu.pybind.utility.Vector3iVector) -> None: ...
    def cluster_connected_triangles(self) -> Any: ...
    def compute_adjacency_list(self) -> Any: ...
    def compute_convex_hull(self) -> Any: ...
    def compute_triangle_normals(self, normalized = ...) -> Any: ...
    def compute_vertex_normals(self, normalized = ...) -> Any: ...
    def create_arrow(self, *args, **kwargs) -> Any: ...
    def create_box(self, *args, **kwargs) -> Any: ...
    def create_cone(self, *args, **kwargs) -> Any: ...
    def create_coordinate_frame(self, *args, **kwargs) -> Any: ...
    def create_cylinder(self, *args, **kwargs) -> Any: ...
    def create_from_oriented_bounding_box(self, *args, **kwargs) -> Any: ...
    def create_from_point_cloud_alpha_shape(self, *args, **kwargs) -> Any: ...
    def create_from_point_cloud_ball_pivoting(self, *args, **kwargs) -> Any: ...
    def create_from_point_cloud_poisson(self, *args, **kwargs) -> Any: ...
    def create_icosahedron(self, *args, **kwargs) -> Any: ...
    def create_mobius(self, *args, **kwargs) -> Any: ...
    def create_octahedron(self, *args, **kwargs) -> Any: ...
    def create_sphere(self, *args, **kwargs) -> Any: ...
    def create_tetrahedron(self, *args, **kwargs) -> Any: ...
    def create_torus(self, *args, **kwargs) -> Any: ...
    @overload
    def crop(self, bounding_box) -> Any: ...
    @overload
    def crop(self, bounding_box) -> Any: ...
    def deform_as_rigid_as_possible(self, constraint_vertex_indices, constraint_vertex_positions, max_iter, energy = ..., smoothed_alpha = ...) -> Any: ...
    def euler_poincare_characteristic(self) -> Any: ...
    def filter_sharpen(self, number_of_iterations = ..., strength = ..., filter_scope = ...) -> Any: ...
    def filter_smooth_laplacian(self, number_of_iterations = ..., lambda_filter = ..., filter_scope = ...) -> Any: ...
    def filter_smooth_simple(self, number_of_iterations = ..., filter_scope = ...) -> Any: ...
    def filter_smooth_taubin(self, number_of_iterations = ..., lambda_filter = ..., mu = ..., filter_scope = ...) -> Any: ...
    def get_non_manifold_edges(self, allow_boundary_edges = ...) -> Any: ...
    def get_non_manifold_vertices(self) -> Any: ...
    def get_self_intersecting_triangles(self) -> Any: ...
    def get_surface_area(self) -> float: ...
    def get_volume(self) -> float: ...
    def has_adjacency_list(self) -> Any: ...
    def has_textures(self) -> Any: ...
    def has_triangle_material_ids(self) -> Any: ...
    def has_triangle_normals(self) -> Any: ...
    def has_triangle_uvs(self) -> Any: ...
    def has_triangles(self) -> Any: ...
    def has_vertex_colors(self) -> Any: ...
    def has_vertex_normals(self) -> Any: ...
    def has_vertices(self) -> Any: ...
    def is_edge_manifold(self, allow_boundary_edges = ...) -> Any: ...
    def is_intersecting(self, arg0) -> Any: ...
    def is_orientable(self) -> Any: ...
    def is_self_intersecting(self) -> Any: ...
    def is_vertex_manifold(self) -> Any: ...
    def is_watertight(self) -> Any: ...
    def merge_close_vertices(self, eps) -> Any: ...
    def normalize_normals(self) -> Any: ...
    def orient_triangles(self) -> Any: ...
    def paint_uniform_color(self, arg0) -> Any: ...
    def remove_degenerate_triangles(self) -> Any: ...
    def remove_duplicated_triangles(self) -> Any: ...
    def remove_duplicated_vertices(self) -> Any: ...
    def remove_non_manifold_edges(self) -> Any: ...
    def remove_triangles_by_index(self, triangle_indices) -> Any: ...
    def remove_triangles_by_mask(self, triangle_mask) -> Any: ...
    def remove_unreferenced_vertices(self) -> Any: ...
    def remove_vertices_by_index(self, vertex_indices) -> Any: ...
    def remove_vertices_by_mask(self, vertex_mask) -> Any: ...
    def sample_points_poisson_disk(self, number_of_points, init_factor = ..., pcl = ..., use_triangle_normal = ...) -> Any: ...
    def sample_points_uniformly(self, number_of_points = ..., use_triangle_normal = ...) -> Any: ...
    def select_by_index(self, indices, cleanup = ...) -> Any: ...
    def simplify_quadric_decimation(self, target_number_of_triangles, maximum_error = ..., boundary_weight = ...) -> Any: ...
    def simplify_vertex_clustering(self, voxel_size, contraction = ...) -> Any: ...
    def subdivide_loop(self, number_of_iterations = ...) -> Any: ...
    def subdivide_midpoint(self, number_of_iterations = ...) -> Any: ...
    def __add__(self, arg0: TriangleMesh) -> TriangleMesh: ...
    def __copy__(self) -> TriangleMesh: ...
    def __deepcopy__(self, arg0: dict) -> TriangleMesh: ...
    def __iadd__(self, arg0: TriangleMesh) -> TriangleMesh: ...

class Voxel:
    color: numpy.ndarray[numpy.float64[3,1]]
    grid_index: numpy.ndarray[numpy.int32[3,1]]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: Voxel) -> None: ...
    @overload
    def __init__(self, grid_index: numpy.ndarray[numpy.int32[3,1]]) -> None: ...
    @overload
    def __init__(self, grid_index: numpy.ndarray[numpy.int32[3,1]], color: numpy.ndarray[numpy.float64[3,1]]) -> None: ...
    def __copy__(self) -> Voxel: ...
    def __deepcopy__(self, arg0: dict) -> Voxel: ...

class VoxelGrid(Geometry3D):
    origin: numpy.ndarray[numpy.float64[3,1]]
    voxel_size: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: VoxelGrid) -> None: ...
    def carve_depth_map(self, depth_map, camera_params, keep_voxels_outside_image = ...) -> Any: ...
    def carve_silhouette(self, silhouette_mask, camera_params, keep_voxels_outside_image = ...) -> Any: ...
    def check_if_included(self, queries) -> Any: ...
    def create_dense(self, *args, **kwargs) -> Any: ...
    def create_from_octree(self, octree) -> Any: ...
    def create_from_point_cloud(self, *args, **kwargs) -> Any: ...
    def create_from_point_cloud_within_bounds(self, *args, **kwargs) -> Any: ...
    def create_from_triangle_mesh(self, *args, **kwargs) -> Any: ...
    def create_from_triangle_mesh_within_bounds(self, *args, **kwargs) -> Any: ...
    def get_voxel(self, point) -> Any: ...
    def get_voxel_bounding_points(self, index) -> Any: ...
    def get_voxel_center_coordinate(self, idx) -> Any: ...
    def get_voxels(self) -> List[Voxel]: ...
    def has_colors(self) -> Any: ...
    def has_voxels(self) -> Any: ...
    def to_octree(self, max_depth) -> Any: ...
    def __add__(self, arg0: VoxelGrid) -> VoxelGrid: ...
    def __copy__(self) -> VoxelGrid: ...
    def __deepcopy__(self, arg0: dict) -> VoxelGrid: ...
    def __iadd__(self, arg0: VoxelGrid) -> VoxelGrid: ...

def get_rotation_matrix_from_axis_angle(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_quaternion(rotation: numpy.ndarray[numpy.float64[4,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_xyz(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_xzy(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_yxz(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_yzx(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_zxy(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
def get_rotation_matrix_from_zyx(rotation: numpy.ndarray[numpy.float64[3,1]]) -> numpy.ndarray[numpy.float64[3,3]]: ...
