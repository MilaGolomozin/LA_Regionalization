"""
la_geodesic_labeling.py
=======================
Left Atrium (LA) surface parcellation via geodesic paths.

Pipeline overview
-----------------
1. Load a labelled NIfTI segmentation and a VTK surface mesh of the LA.
2. Compute the world-space centre-of-mass (ostium) for each pulmonary vein
   (PV), the left atrial appendage (LAA), and the mitral valve (MV).
3. Snap every ostium seed to the nearest vertex on the LA surface mesh.
4. Compute Dijkstra geodesic paths between selected pairs of seeds to form
   anatomical boundary lines on the mesh.
5. Use BFS region-growing (bounded by the path vertices) to assign each mesh
   vertex to a distinct anatomical region.
6. Save outputs:
     - seeds_new.vtk          : the projected seed points
     - geodesic_paths.vtk     : all geodesic paths merged into one file
     - mesh_with_regions.vtk  : LA mesh with per-vertex RegionID scalar

Dependencies: nibabel, numpy, scipy, vtk, edt (euclidean distance transform)
"""

import nibabel as nib
import numpy as np
import vtk
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from collections import deque, defaultdict

# ─────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────

# NIfTI segmentation: each voxel holds an integer label identifying the
# anatomical structure it belongs to.
nii  = nib.load("/8_with_mitral_valve_new.nii.gz")
data = nii.get_fdata()
affine = nii.affine          # 4×4 voxel-to-world transform (RAS space)

# Label values in the segmentation volume
LA   = 2   # left atrium body
LSPV, LIPV, RSPV, RIPV = 10, 11, 12, 13  # four pulmonary veins
LAA  = 8   # left atrial appendage
MV   = 20  # mitral valve

# VTK surface mesh of the LA (vertex coordinates are in LPS space)
la_pd_reader = vtk.vtkPolyDataReader()
la_pd_reader.SetFileName("/8_surface_with_labels.vtk")
la_pd_reader.Update()
la_pd = la_pd_reader.GetOutput()


# ─────────────────────────────────────────────────────────────────
# 2.  Compute ostium centres of mass (in RAS world space)
# ─────────────────────────────────────────────────────────────────

def get_ostium_com(data, pv_label, la_label, affine):
    """
    Return the world-space centre of mass of the interface between a
    pulmonary vein / appendage and the LA body.

    The interface is found by dilating the LA mask by one voxel and
    intersecting the result with the PV mask.  This gives a thin ring of
    voxels at the ostium opening.

    Parameters
    ----------
    data      : 3-D integer array  – segmentation volume
    pv_label  : int                – label of the PV (or LAA)
    la_label  : int                – label of the LA body
    affine    : (4,4) float array  – voxel-to-world transform

    Returns
    -------
    com_world : (3,) float array   – centre of mass in world (RAS) coords
    """
    pv_mask = (data == pv_label)
    la_mask = (data == la_label)

    # Dilate LA by 1 voxel so it overlaps slightly into the PV opening
    la_dilated = binary_dilation(la_mask)
    interface  = pv_mask & la_dilated           # ostium voxels

    coords_vox = np.argwhere(interface)         # shape (N, 3)
    com_vox    = coords_vox.mean(axis=0)        # voxel-space centre of mass

    # Apply the affine to convert to world (RAS) coordinates
    com_world = affine @ np.append(com_vox, 1.0)
    return com_world[:3]


# Ostium seeds for PVs and LAA (RAS world space)
s_lspv = get_ostium_com(data, LSPV, LA, affine)
s_lipv = get_ostium_com(data, LIPV, LA, affine)
s_rspv = get_ostium_com(data, RSPV, LA, affine)
s_ripv = get_ostium_com(data, RIPV, LA, affine)
s_laa  = get_ostium_com(data, LAA,  LA, affine)


# ─────────────────────────────────────────────────────────────────
# 3.  Derive four MV anchor points closest to each PV ostium
# ─────────────────────────────────────────────────────────────────

# Convert all MV voxels to world (RAS) coordinates
mv_vox   = np.argwhere(data == MV)
ones     = np.ones((len(mv_vox), 1))
mv_world = (affine @ np.hstack([mv_vox, ones]).T).T[:, :3]

# Build a KD-tree for fast nearest-neighbour queries on the MV point cloud
mv_tree = cKDTree(mv_world)

# For each PV, find the closest MV point and apply a small manual offset
# (these offsets empirically place the anchor on the annulus ridge).
_, idx = mv_tree.query(s_rspv); mv1 = mv_world[idx] + [0,  0, -15]
_, idx = mv_tree.query(s_ripv); mv2 = mv_world[idx] + [0,  5, -50]
_, idx = mv_tree.query(s_lipv); mv3 = mv_world[idx]
_, idx = mv_tree.query(s_lspv); mv4 = mv_world[idx]


# ─────────────────────────────────────────────────────────────────
# 4.  Convert seeds from RAS (NIfTI) → LPS (VTK mesh) space
# ─────────────────────────────────────────────────────────────────

def ras_to_lps(pt):
    """Flip X and Y axes to convert RAS coordinates to LPS coordinates."""
    return np.array([-pt[0], -pt[1], pt[2]])


s_lspv = ras_to_lps(s_lspv)
s_lipv = ras_to_lps(s_lipv)
s_rspv = ras_to_lps(s_rspv)
s_ripv = ras_to_lps(s_ripv)
s_laa  = ras_to_lps(s_laa)
mv1    = ras_to_lps(mv1)
mv2    = ras_to_lps(mv2)
mv3    = ras_to_lps(mv3)
mv4    = ras_to_lps(mv4)

# Ordered list of seeds and their display names
seeds_world = [s_lspv, s_lipv, s_rspv, s_ripv, s_laa, mv1, mv2, mv3, mv4]
seed_names  = ["LSPV", "LIPV", "RSPV", "RIPV", "LAA", "MV1", "MV2", "MV3", "MV4"]


# ─────────────────────────────────────────────────────────────────
# 5.  Project seeds onto the nearest mesh vertex
# ─────────────────────────────────────────────────────────────────

def project_to_mesh(world_pt, locator, mesh):
    """
    Find the mesh vertex closest to *world_pt* using a pre-built
    vtkPointLocator and return its index and coordinates.
    """
    idx = locator.FindClosestPoint(world_pt.tolist())
    return idx, np.array(mesh.GetPoint(idx))


# Build a spatial index over all mesh vertices
locator = vtk.vtkPointLocator()
locator.SetDataSet(la_pd)
locator.BuildLocator()

# Create a VTK PolyData that stores the projected seed positions
seed_pd    = vtk.vtkPolyData()
pts        = vtk.vtkPoints()
region_ids = vtk.vtkIntArray()
region_ids.SetName("SeedID")

seed_indices = {}   # name → vertex index on the mesh

for i, (name, pt) in enumerate(zip(seed_names, seeds_world)):
    idx, proj = project_to_mesh(pt, locator, la_pd)
    seed_indices[name] = idx
    pts.InsertNextPoint(proj[0], proj[1], proj[2])
    region_ids.InsertNextValue(i)
    print(f"{name}: vertex {idx} @ {proj.round(2)}")

seed_pd.SetPoints(pts)
seed_pd.GetPointData().SetScalars(region_ids)

# Each seed is a VTK vertex cell (required so the file is renderable)
verts = vtk.vtkCellArray()
for i in range(pts.GetNumberOfPoints()):
    verts.InsertNextCell(1)
    verts.InsertCellPoint(i)
seed_pd.SetVerts(verts)

print(f"\nSeed polydata: {seed_pd.GetNumberOfPoints()} points, "
      f"{seed_pd.GetNumberOfCells()} cells")

# Write seeds to disk
writer = vtk.vtkPolyDataWriter()
writer.SetInputData(seed_pd)
writer.SetFileName("seeds_new.vtk")
writer.SetFileTypeToASCII()
writer.Write()
print("Saved seeds_new.vtk")


# ─────────────────────────────────────────────────────────────────
# 6.  Compute geodesic paths between seed pairs
# ─────────────────────────────────────────────────────────────────

def compute_geodesic(mesh, start_idx, end_idx):
    """
    Compute the shortest path along mesh edges (Dijkstra) between two
    vertex indices.

    Returns
    -------
    path_pd   : vtkPolyData  – polyline of the path
    path_ids  : list[int]    – ordered vertex indices along the path
    """
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(mesh)
    dijkstra.SetStartVertex(start_idx)
    dijkstra.SetEndVertex(end_idx)
    dijkstra.Update()

    path_pd = dijkstra.GetOutput()

    # GetIdList() API differs between VTK versions; handle both
    try:
        id_list  = dijkstra.GetIdList()                     # newer VTK
        path_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
    except TypeError:
        id_list  = vtk.vtkIdList()                          # older VTK
        dijkstra.GetIdList(id_list)
        path_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]

    return path_pd, path_ids


# Anatomical boundary lines connecting pairs of seeds.
# These paths together form a closed boundary network that partitions the
# LA surface into discrete anatomical regions.
paths_to_compute = {
    "s1":  ("RSPV", "RIPV"),   # right PV ridge (superior–inferior)
    "s2":  ("LIPV", "RIPV"),   # posterior wall bottom (left–right)
    "s3":  ("LSPV", "LIPV"),   # left PV ridge (superior–inferior)
    "s4":  ("LSPV", "RSPV"),   # roof line
    "s5":  ("RSPV", "MV1"),    # right superior PV to MV anchor
    "s6":  ("RIPV", "MV2"),    # right inferior PV to MV anchor
    "s7":  ("LIPV", "MV3"),    # left inferior PV to MV anchor
    "s8":  ("MV3",  "MV4"),    # MV isthmus (inferior)
    "s9":  ("LAA",  "LSPV"),   # LAA to left superior PV
    "s10": ("LAA",  "MV4"),    # LAA to MV anchor
    "s11": ("MV1",  "MV4"),    # MV roof (superior)
    "s12": ("MV1",  "MV2"),    # right MV isthmus
    "s13": ("MV3",  "MV2"),    # left MV isthmus
}

all_path_polydata = {}   # name → vtkPolyData (polyline)
all_path_ids      = {}   # name → list of vertex indices

for path_name, (a, b) in paths_to_compute.items():
    path_pd, path_ids = compute_geodesic(la_pd, seed_indices[a], seed_indices[b])
    all_path_polydata[path_name] = path_pd
    all_path_ids[path_name]      = path_ids
    print(f"{path_name} ({a} → {b}): {len(path_ids)} vertices")


# ─────────────────────────────────────────────────────────────────
# 7.  Save geodesic paths to a single VTK file
# ─────────────────────────────────────────────────────────────────

def save_paths_as_vtk(all_path_polydata, filename="geodesic_paths.vtk"):
    """
    Merge all geodesic path polylines into a single VTK PolyData file.
    Each point carries an integer 'PathID' scalar so that individual paths
    can be distinguished in downstream tools (e.g. 3D Slicer).
    """
    append = vtk.vtkAppendPolyData()

    for path_name, path_pd in all_path_polydata.items():
        path_num      = int(path_name[1:])   # "s1" → 1, "s12" → 12
        path_id_array = vtk.vtkIntArray()
        path_id_array.SetName("PathID")
        for _ in range(path_pd.GetNumberOfPoints()):
            path_id_array.InsertNextValue(path_num)
        path_pd.GetPointData().SetScalars(path_id_array)
        append.AddInputData(path_pd)

    append.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(append.GetOutput())
    writer.SetFileName(filename)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f"Saved {filename}")


save_paths_as_vtk(all_path_polydata, "geodesic_paths.vtk")


# ─────────────────────────────────────────────────────────────────
# 8.  Region growing: label LA mesh vertices by anatomical region
# ─────────────────────────────────────────────────────────────────

def build_vertex_adjacency(mesh_pd):
    """
    Build an adjacency list (dict of sets) from mesh triangle cells.
    Two vertices are adjacent if they share a cell edge.
    """
    adjacency = defaultdict(set)
    for i in range(mesh_pd.GetNumberOfCells()):
        cell = mesh_pd.GetCell(i)
        ids  = cell.GetPointIds()
        n    = ids.GetNumberOfIds()
        for j in range(n):
            v1 = ids.GetId(j)
            v2 = ids.GetId((j + 1) % n)
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    return adjacency


def label_regions(mesh_pd, all_path_ids, filename="mesh_with_regions.vtk"):
    """
    Assign each non-boundary mesh vertex a unique integer RegionID using
    breadth-first search (BFS).  Path vertices act as hard boundaries that
    prevent BFS from crossing between regions.

    The resulting scalar array 'RegionID' encodes:
        0  – vertex lies on a geodesic boundary path
        1+ – distinct anatomical region index

    The labelled mesh is written to *filename*.
    """
    n_pts = mesh_pd.GetNumberOfPoints()

    # Collect all vertices that form the boundary network
    boundary_vertices = set()
    for ids in all_path_ids.values():
        boundary_vertices.update(ids)

    adjacency     = build_vertex_adjacency(mesh_pd)
    region_labels = [-1] * n_pts   # -1 = unlabelled
    current_region = 1

    # BFS from every unlabelled, non-boundary vertex
    for v in range(n_pts):
        if v in boundary_vertices or region_labels[v] != -1:
            continue

        queue = deque([v])
        region_labels[v] = current_region

        while queue:
            curr = queue.popleft()
            for neighbour in adjacency[curr]:
                if neighbour in boundary_vertices:
                    continue
                if region_labels[neighbour] == -1:
                    region_labels[neighbour] = current_region
                    queue.append(neighbour)

        current_region += 1

    print(f"Detected {current_region - 1} anatomical regions")

    # Pack region labels into a VTK integer array
    region_array = vtk.vtkIntArray()
    region_array.SetName("RegionID")
    region_array.SetNumberOfValues(n_pts)
    for i in range(n_pts):
        region_array.SetValue(i, 0 if i in boundary_vertices else region_labels[i])

    # Attach the array to a copy of the mesh and save
    out_mesh = vtk.vtkPolyData()
    out_mesh.DeepCopy(mesh_pd)
    out_mesh.GetPointData().AddArray(region_array)
    out_mesh.GetPointData().SetActiveScalars("RegionID")

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(out_mesh)
    writer.SetFileName(filename)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f"Saved {filename}")


label_regions(la_pd, all_path_ids, "/8_regions_with_labels.vtk")


# ─────────────────────────────────────────────────────────────────
# 9.  Interactive visualisation (optional, opens a VTK render window)
# ─────────────────────────────────────────────────────────────────

def visualize_seeds_on_mesh(mesh_pd, seed_pd, seed_names, path_polydatas):
    """
    Render the LA mesh, geodesic paths (as colour-coded tubes), and seed
    spheres with text labels in an interactive VTK window.

    Controls: left-click + drag to rotate, scroll to zoom, right-click + drag
    to pan (standard vtkInteractorStyleTrackballCamera).
    """
    renderer     = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("Seeds + Geodesic Paths on LA Mesh")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # -- Mesh (semi-transparent grey) --
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh_pd)
    mesh_mapper.ScalarVisibilityOff()
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    mesh_actor.GetProperty().SetOpacity(0.4)
    renderer.AddActor(mesh_actor)

    # -- Geodesic paths as colour-coded tubes --
    path_colors = {
        "s1":  (1.0, 0.0, 0.0),   # red
        "s2":  (1.0, 0.5, 0.0),   # orange
        "s3":  (1.0, 1.0, 0.0),   # yellow
        "s4":  (0.0, 1.0, 0.0),   # green
        "s5":  (0.0, 1.0, 0.8),   # teal
        "s6":  (0.0, 0.5, 1.0),   # light blue
        "s7":  (0.0, 0.0, 1.0),   # blue
        "s8":  (0.6, 0.0, 1.0),   # purple
        "s9":  (1.0, 0.0, 0.8),   # pink
        "s10": (1.0, 0.0, 0.0),
        "s11": (1.0, 0.5, 0.0),
        "s12": (1.0, 1.0, 0.0),
        "s13": (0.0, 0.5, 1.0),
    }

    for path_name, path_pd in path_polydatas.items():
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(path_pd)
        tube.SetRadius(0.8)
        tube.SetNumberOfSides(12)
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*path_colors[path_name])
        actor.GetProperty().SetSpecular(0.3)
        renderer.AddActor(actor)

        # Float the path label at the midpoint of the polyline
        n = path_pd.GetNumberOfPoints()
        if n > 0:
            mid_pt = path_pd.GetPoint(n // 2)
            label  = vtk.vtkBillboardTextActor3D()
            label.SetInput(path_name)
            label.SetPosition(mid_pt[0], mid_pt[1] + 3, mid_pt[2])
            label.GetTextProperty().SetFontSize(12)
            label.GetTextProperty().SetColor(*path_colors[path_name])
            label.GetTextProperty().BoldOn()
            renderer.AddActor(label)

    # -- Seeds as coloured spheres with labels --
    seed_colors = [
        (1.0, 0.2, 0.2),  # LSPV  – red
        (1.0, 0.6, 0.0),  # LIPV  – orange
        (0.2, 0.8, 0.2),  # RSPV  – green
        (0.5, 0.0, 0.8),  # RIPV  – purple
        (0.0, 0.8, 0.8),  # LAA   – cyan
        (1.0, 1.0, 1.0),  # MV1   – white
        (0.8, 0.8, 0.8),  # MV2   – light grey
        (0.6, 0.6, 0.6),  # MV3   – mid grey
        (0.4, 0.4, 0.4),  # MV4   – dark grey
    ]

    for i in range(seed_pd.GetNumberOfPoints()):
        pt = seed_pd.GetPoint(i)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pt)
        sphere.SetRadius(3.0)
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*seed_colors[i])
        renderer.AddActor(actor)

        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(seed_names[i])
        label.SetPosition(pt[0], pt[1] + 4, pt[2])
        label.GetTextProperty().SetFontSize(14)
        label.GetTextProperty().SetColor(*seed_colors[i])
        label.GetTextProperty().BoldOn()
        renderer.AddActor(label)

    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


# Comment this line out if running headless (e.g. on a cluster).
visualize_seeds_on_mesh(la_pd, seed_pd, seed_names, all_path_polydata)