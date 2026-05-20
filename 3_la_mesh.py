"""
3_la_mesh.py
================
Pipeline for converting raw medical image segmentations into clean, labeled 3D surface meshes.

The three stages run in order:

1. prepare_meshes_from_segmentations()
   Reads a multi-label NIfTI segmentation file (.nii.gz), extracts a binary mask for
   the left atrium (and optionally other structures), saves the mask, and converts it
   to a raw VTK surface mesh using marching-cubes style extraction.

2. decimate_and_smooth()
   Takes the raw surface mesh and cleans it up for practical use:
     - Keeps only the largest connected region (discards floating fragments)
     - Fills holes in the surface
     - Triangulates all polygons
     - Removes duplicate/degenerate geometry
     - Runs two rounds of decimation (each removing 50% of triangles) interleaved
       with constrained smoothing, resulting in a mesh at ~25% of its original
       polygon count that is smooth but geometrically faithful
     - Computes surface normals for rendering

3. propagate_labels_to_mesh()
   Re-associates anatomical labels with the decimated mesh. For each mesh vertex,
   samples a 5×5×5 voxel neighbourhood in the original segmentation image and assigns
   the majority label to that vertex as a scalar. The result is a mesh whose vertices
   are colour-coded by anatomical region, ready for inspection or further analysis.
"""

import numpy as np
import processing_tools as pt
import SimpleITK as sitk
import vtk


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

SEG_IN          = "/8_with_mitral_valve_new.nii.gz"
SEG_OUT         = "/8_la_laa_pv.nii.gz"
SURFACE_RAW     = "/8_raw_surface.vtk"
SURFACE_DECIMATED = "/8_decimated_surface.vtk"
SURFACE_LABELED = "/8_surface_with_labels.vtk"

# Label values present in the segmentation image
LA_LABEL   = 2
# LAA_LABEL  = 8
# LSPV_LABEL = 10
# LIPV_LABEL = 11
# RSPV_LABEL = 12
# RIPV_LABEL = 13

LEGAL_LABELS = [LA_LABEL]
# LEGAL_LABELS = [LA_LABEL, LAA_LABEL, LSPV_LABEL, LIPV_LABEL, RSPV_LABEL, RIPV_LABEL]


# ---------------------------------------------------------------------------
# Stage 1: Extract binary mask and raw surface
# ---------------------------------------------------------------------------

def prepare_meshes_from_segmentations():
    """
    Read a multi-label segmentation, isolate the structures of interest,
    save the resulting binary mask, and extract a raw surface mesh from it.
    """
    print(f"Reading segmentation: {SEG_IN}")

    try:
        label_img = sitk.ReadImage(SEG_IN)
    except RuntimeError as e:
        print(f"Error reading {SEG_IN}: {e}")
        return

    print(f"Image size: {label_img.GetSize()}")

    label_img_np = sitk.GetArrayFromImage(label_img)
    # Note: SimpleITK uses (x, y, z) but NumPy uses (z, y, x), so axes are transposed.

    unique_labels = np.unique(label_img_np)
    print(f"Labels found in image: {unique_labels}")

    # Build a binary mask for the structures of interest
    combined_mask = (label_img_np == LA_LABEL)
    # Uncomment to include additional structures:
    # combined_mask = (
    #     (label_img_np == LA_LABEL)
    #     | (label_img_np == LAA_LABEL)
    #     | (label_img_np == LSPV_LABEL)
    #     | (label_img_np == LIPV_LABEL)
    #     | (label_img_np == RSPV_LABEL)
    #     | (label_img_np == RIPV_LABEL)
    # )

    # Convert back to a SimpleITK image, preserving spatial metadata
    combined_mask_img = sitk.GetImageFromArray(combined_mask.astype(np.uint8))
    combined_mask_img.CopyInformation(label_img)
    sitk.WriteImage(combined_mask_img, SEG_OUT)
    print(f"Saved combined mask to {SEG_OUT}")

    print("Extracting surface mesh...")
    pt.convert_label_map_to_surface_file(SEG_OUT, SURFACE_RAW)
    print(f"Saved raw surface to {SURFACE_RAW}")


# ---------------------------------------------------------------------------
# Stage 2: Decimate and smooth the raw surface
# ---------------------------------------------------------------------------

def _build_decimator(input_data, target_reduction=0.50, max_error=2.0):
    """Create and run a vtkDecimatePro filter on *input_data*."""
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(input_data)
    decimate.SetTargetReduction(target_reduction)
    decimate.SplittingOn()
    decimate.SetMaximumError(max_error)
    decimate.PreserveTopologyOn()
    decimate.Update()
    return decimate


def _build_smoother(input_data, iterations=1000, relaxation=0.01, constraint_dist=5.0):
    """Create and run a vtkConstrainedSmoothingFilter on *input_data*."""
    smoother = vtk.vtkConstrainedSmoothingFilter()
    smoother.SetInputData(input_data)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetRelaxationFactor(relaxation)
    smoother.SetConstraintDistance(constraint_dist)
    smoother.SetConstraintStrategyToConstraintDistance()
    smoother.Update()
    return smoother


def decimate_and_smooth():
    """
    Clean up the raw surface mesh:
      1. Keep only the largest connected region.
      2. Fill holes up to 1000 sq-units in size.
      3. Triangulate all polygons and remove degenerate geometry.
      4. Two rounds of 50% decimation, each followed by constrained smoothing.
      5. Compute point and cell normals.
      6. Save the result.
    """
    # --- Load ---
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(SURFACE_RAW)
    reader.Update()

    # --- Keep only the largest connected region ---
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(reader.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    print(f"Connected regions found: {conn.GetNumberOfExtractedRegions()}")

    # --- Fill holes ---
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # --- Ensure all faces are triangles ---
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    # --- Remove duplicates and degenerate elements ---
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    # --- Round 1: decimate then smooth ---
    dec1    = _build_decimator(cleaner.GetOutput())
    smooth1 = _build_smoother(dec1.GetOutput())

    # --- Round 2: decimate then smooth ---
    dec2    = _build_decimator(smooth1.GetOutput())
    smooth2 = _build_smoother(dec2.GetOutput())

    # --- Compute surface normals for rendering ---
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth2.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()   # Shared vertices get averaged normals → smooth shading
    normals.Update()

    # --- Save ---
    print(f"Saving decimated surface to {SURFACE_DECIMATED}")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(SURFACE_DECIMATED)
    writer.SetInputData(normals.GetOutput())
    writer.Write()


# ---------------------------------------------------------------------------
# Stage 3: Propagate anatomical labels onto the mesh vertices
# ---------------------------------------------------------------------------

def get_connected_vertices(mesh, point_idx):
    """
    Return the indices of all vertices directly connected to *point_idx* by an edge.

    Parameters
    ----------
    mesh      : vtkPolyData
    point_idx : int  — index of the query vertex

    Returns
    -------
    list[int]
    """
    connected = set()

    cell_ids = vtk.vtkIdList()
    mesh.GetPointCells(point_idx, cell_ids)

    for i in range(cell_ids.GetNumberOfIds()):
        cell = mesh.GetCell(cell_ids.GetId(i))
        for e in range(cell.GetNumberOfEdges()):
            edge      = cell.GetEdge(e)
            pt_ids    = edge.GetPointIds()
            id0, id1  = pt_ids.GetId(0), pt_ids.GetId(1)
            if id0 == point_idx:
                connected.add(int(id1))
            elif id1 == point_idx:
                connected.add(int(id0))

    return list(connected)


def propagate_labels_to_mesh():
    """
    Paint anatomical labels from the segmentation image onto the decimated mesh.

    For every mesh vertex, sample the original segmentation in a 5×5×5 voxel
    neighbourhood (radius = 2 voxels) and assign the most common valid label as
    a scalar value on that vertex.  Vertices with no valid neighbours are assigned
    label 0 (background).

    The output mesh can be colour-mapped by label in any VTK-compatible viewer.
    """
    SEARCH_RADIUS = 2  # voxels in each direction → (2r+1)³ = 125 samples

    # --- Load mesh ---
    print(f"Reading surface mesh: {SURFACE_DECIMATED}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(SURFACE_DECIMATED)
    reader.Update()
    label_surf = reader.GetOutput()
    n_points   = label_surf.GetNumberOfPoints()

    # --- Load segmentation ---
    print(f"Reading segmentation: {SEG_IN}")
    try:
        label_img = sitk.ReadImage(SEG_IN)
    except RuntimeError as e:
        print(f"Error reading {SEG_IN}: {e}")
        return
    print(f"Image size: {label_img.GetSize()}")

    # Transpose from (z, y, x) to (x, y, z) to align with TransformPhysicalPointToIndex
    label_img_np = sitk.GetArrayFromImage(label_img).transpose(2, 1, 0)
    shape        = label_img_np.shape

    # --- Sample labels per vertex ---
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    offsets = range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)

    for i in range(n_points):
        if i % 1000 == 0:
            print(f"  Processing vertex {i}/{n_points}")

        p     = label_surf.GetPoint(i)
        p_idx = np.asarray(label_img.TransformPhysicalPointToIndex(p))

        label_samples = []
        for d in offsets:
            for e in offsets:
                for f in offsets:
                    sample = p_idx + np.array([d, e, f])
                    in_bounds = (
                        0 <= sample[0] < shape[0]
                        and 0 <= sample[1] < shape[1]
                        and 0 <= sample[2] < shape[2]
                    )
                    if in_bounds:
                        val = label_img_np[sample[0], sample[1], sample[2]]
                        if val in LEGAL_LABELS:
                            label_samples.append(val)

        if label_samples:
            labels, counts  = np.unique(label_samples, return_counts=True)
            majority_label  = labels[np.argmax(counts)]
            scalars.InsertNextValue(majority_label)
        else:
            scalars.InsertNextValue(0)
            print(f"  Warning: no valid label samples for vertex {i} at {p}")

    # Attach scalar labels to the mesh and save
    label_surf.GetPointData().SetScalars(scalars)

    print(f"Saving labeled surface to {SURFACE_LABELED}")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(SURFACE_LABELED)
    writer.SetInputData(label_surf)
    writer.Write()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_meshes_from_segmentations()
    decimate_and_smooth()
    propagate_labels_to_mesh()