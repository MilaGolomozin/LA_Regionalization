

import numpy as np
import nibabel as nib
import edt
from scipy import ndimage


# ── Default label configuration ───────────────────────────────────────────────
la_label = 2
pv_label = 10

# ── Default stopping criteria ─────────────────────────────────────────────────
max_la_growth = 0.80
min_pv_remaining = 0.30
target_pv_components = 4

# ── Balance check: how similar must the PV components be? ─────────────────────
# A valid separation requires that every component is at least this fraction
# of the mean component size.  0.25 means no component can be < 25% of average.
min_size_ratio = 0.25


#Loading the nifti file

img = nib.load("ImageCAS-STACOM2025-02-10-2025/segmentations/7.img.nii.gz")
seg_data = img.get_fdata().astype(np.int32)
    

#Helper function to save a nifti file

def save_nifti(path, data, ref_img):
    out = nib.Nifti1Image(data.astype(np.int32), ref_img.affine, ref_img.header)
    nib.save(out, path)
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def clean_spurious_pv_components(seg, la_label, pv_label, min_size_ratio=50):
    """
    Remove PV connected components that are tiny compared to the LA volume.
    A component is deleted if:  component_size < (la_size / min_size_ratio)
    """
    la_size = int((seg == la_label).sum())
    size_threshold = la_size / min_size_ratio

    pv_mask = seg == pv_label
    labeled, n = ndimage.label(pv_mask)

    print(f"\nPre-cleanup: {n} PV connected components found")
    print(f"  LA size          : {la_size} voxels")
    print(f"  Min allowed size : {int(size_threshold)} voxels  (LA / {min_size_ratio})")

    seg_clean = seg.copy()
    removed = 0
    for comp_id in range(1, n + 1):
        mask = labeled == comp_id
        size = int(mask.sum())
        if size < size_threshold:
            seg_clean[mask] = 0
            print(f"  Removed component {comp_id}: {size} voxels (below threshold)")
            removed += 1
        else:
            print(f"  Kept    component {comp_id}: {size} voxels")

    print(f"Post-cleanup: {n - removed} PV components remaining ({removed} removed)\n")
    return seg_clean, removed


def get_voxel_sizes(img):
    zooms = img.header.get_zooms()
    return tuple(float(z) for z in zooms[:3])


# ─────────────────────────────────────────────────────────────────────────────
# Component helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_pv_components(pv_mask):
    """
    Count PV components after removing small ones.

    Small components are deleted directly from `seg`.
    Returns:
        n_components (int)
        labeled_array (np.ndarray)
    """
    min_size=100

    # First labeling
    labeled, n = ndimage.label(pv_mask)
    sizes = component_sizes(labeled, n)

    # Find small components
    small_labels = [i + 1 for i, s in enumerate(sizes) if s < min_size]

    # Remove them from segmentation
    if len(small_labels) > 0:
        seg[np.isin(labeled, small_labels)] = 0

    # Recompute after cleanup
    pv_mask = seg == pv_label
    labeled_clean, n_clean = ndimage.label(pv_mask)

    return n_clean, labeled_clean



def component_sizes(labeled, n):
    """Return a list of voxel counts for each component label 1..n."""
    return [int((labeled == i).sum()) for i in range(1, n + 1)]



def is_balanced_separation(labeled, n, min_ratio=0.15):
    """
    True when all n components are approximately the same size.

    Criterion: every component must be at least `min_ratio` × mean_size.
    This rejects splits where one 'component' is a tiny stray blob.
    """
    if n != 4:
        return False
    sizes = component_sizes(labeled, n)
    mean_size = np.mean(sizes)
    return all(s >= min_ratio * mean_size for s in sizes)

#─────────────────────────────────────────────────────────────────────────────
# Post-processing: relabel individual PVs
# ─────────────────────────────────────────────────────────────────────────────


# ── Fixed PV label assignments (always consistent across scans) ───────────────
PV_NAME_TO_LABEL = {
    "LSPV": 10,   # Left Superior Pulmonary Vein
    "LIPV": 11,   # Left Inferior Pulmonary Vein
    "RSPV": 12,   # Right Superior Pulmonary Vein
    "RIPV": 13,   # Right Inferior Pulmonary Vein
}
PV_LABEL_TO_NAME = {v: k for k, v in PV_NAME_TO_LABEL.items()}


def classify_pv_by_anatomy(centroid_voxel, affine):
    """
    Given a voxel-space centroid and the image affine, classify the PV as
    LSPV / LIPV / RSPV / RIPV.

    Steps:
      1. Convert centroid from voxel → RAS world coordinates using the affine.
      2. RAS X: positive = Right patient side, negative = Left patient side.
      3. RAS Z: larger value = Superior, smaller = Inferior.

    Returns one of: 'LSPV', 'LIPV', 'RSPV', 'RIPV'
    """
    # Homogeneous voxel coordinate → world (RAS)
    vox_h = np.array([*centroid_voxel, 1.0])
    ras = affine @ vox_h          # shape (4,)
    x_ras, _, z_ras = ras[0], ras[1], ras[2]

    side      = "R" if x_ras > 0 else "L"
    position  = "S" if z_ras < 0 else "I"   # will refine below with relative Z

    # Return provisional label — will be refined relatively after all centroids known
    return side, position, z_ras, x_ras


def relabel_pv_components(seg, pv_label, affine):
    """
    Relabel each PV connected component with a fixed anatomical label:
        LSPV = 10, LIPV = 11, RSPV = 12, RIPV = 13

    Classification is done by converting each component's centroid from
    voxel space to RAS world coordinates via the affine matrix:
        - X > 0  → Right side of patient;  X < 0 → Left
        - Relative Z among same-side PVs → Superior (higher Z) vs Inferior

    This means the same anatomical PV always gets the same integer label
    regardless of which scan is processed.
    """
    pv_mask = seg == pv_label
    labeled, n = ndimage.label(pv_mask)

    if n == 0:
        print("No PV components found.")
        return seg.copy()

    # ── 1. Compute centroids in voxel space ──────────────────────────────────
    centroids = {}
    sizes = {}
    for comp_id in range(1, n + 1):
        coords = np.argwhere(labeled == comp_id)
        centroids[comp_id] = coords.mean(axis=0)       # (i, j, k) voxel centroid
        sizes[comp_id]     = len(coords)

    # ── 2. Convert to RAS and get side + z for each component ────────────────
    ras_info = {}   # comp_id -> {'side': 'L'/'R', 'z': float, 'x': float}
    for comp_id, centroid in centroids.items():
        side, _, z_ras, x_ras = classify_pv_by_anatomy(centroid, affine)
        ras_info[comp_id] = {'side': side, 'z': z_ras, 'x': x_ras}

    # ── 3. Within each side, assign Superior/Inferior by relative Z ──────────
    #    (avoids hard-coding a Z threshold that varies across patients)
    assignment = {}   # comp_id -> PV name string
    for side in ('L', 'R'):
        same_side = [cid for cid, info in ras_info.items() if info['side'] == side]

        if len(same_side) == 0:
            print(f"  WARNING: No components found on {side} side.")
        elif len(same_side) == 1:
            # Only one PV on this side — default to Superior (common left trunk case)
            cid = same_side[0]
            name = f"{side}SPV"
            assignment[cid] = name
            print(f"  WARNING: Only 1 component on {side} side → assigned as {name}")
        else:
            # Sort by Z descending: highest Z = Superior
            same_side_sorted = sorted(same_side, key=lambda c: ras_info[c]['z'], reverse=True)
            positions = ['S', 'I'] + ['I'] * (len(same_side) - 2)  # S, I, I, ... for extras
            for pos, cid in zip(positions, same_side_sorted):
                assignment[cid] = f"{side}{pos}PV"

    # ── 4. Build the relabeled volume ────────────────────────────────────────
    result = seg.copy()
    result[pv_mask] = 0    # clear all PV voxels first

    print(f"\nRelabeled {n} PV components:")
    print(f"  {'Component':>10} | {'Name':>6} | {'Label':>6} | {'Voxels':>8} | {'RAS X':>8} | {'RAS Z':>8}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    used_names = set()
    for comp_id in range(1, n + 1):
        name = assignment.get(comp_id, f"PV_extra_{comp_id}")

        if name in PV_NAME_TO_LABEL:
            new_label = PV_NAME_TO_LABEL[name]
        else:
            # Fallback for unexpected extra components
            new_label = pv_label + 10 + comp_id
            print(f"  WARNING: Unexpected PV '{name}' — assigning fallback label {new_label}")

        if name in used_names:
            print(f"  WARNING: Duplicate assignment for '{name}' — two components mapped to same anatomy!")
        used_names.add(name)

        result[labeled == comp_id] = new_label

        info = ras_info[comp_id]
        print(f"  {comp_id:>10} | {name:>6} | {new_label:>6} | {sizes[comp_id]:>8} | {info['x']:>8.1f} | {info['z']:>8.1f}")

    # ── 5. Summary legend ────────────────────────────────────────────────────
    print(f"\n  Label legend (fixed across all scans):")
    for name, lbl in sorted(PV_NAME_TO_LABEL.items(), key=lambda x: x[1]):
        status = "assigned" if name in used_names else "NOT FOUND in this scan"
        print(f"    {lbl:>3} = {name}  ({status})")

    return result


# THE PROCESS

# ── Step 0: clean tiny spurious blobs ────────────────────────────────────
seg, _ = clean_spurious_pv_components(seg_data, la_label, pv_label)

la_size_0 = int((seg == la_label).sum())
pv_size_0 = int((seg == pv_label).sum())

if la_size_0 == 0:
    raise ValueError(f"No LA voxels found with label {la_label}. Check --la-label.")
if pv_size_0 == 0:
    raise ValueError(f"No PV voxels found with label {pv_label}. Check --pv-label.")

n0, _ = count_pv_components(seg == pv_label)
print(f"Initial state:")
print(f"  LA voxels     : {la_size_0}")
print(f"  PV voxels     : {pv_size_0}")
print(f"  PV components : {n0}")
print()


#Now we know the total number of voxels for the 4 pvs (of course this includes the oversegmented LA)
#We can therefore define an approximate size for the target PVs and compare the results against it

target_pv_size= pv_size_0/4

# Reassign ~0.1 % of original PV size per iteration (min 1 voxel)
step_size = max(1, int(pv_size_0 * 0.001))
print(f"Step size per iteration: {step_size} voxels\n")


log_rows = []
iteration = 0
best_n_components = n0
separation_iteration = None

while True:
    iteration += 1

    la_mask = seg == la_label
    pv_mask = seg == pv_label

    # Compute LA center of mass (doing it in the loop as the LA changes every iteration since we are reassigning pixels)
    la_coords = np.argwhere(la_mask) # this is a new line
    la_com = la_coords.mean(axis=0) # this too

    la_size = int(la_mask.sum())
    pv_size = int(pv_mask.sum())
    la_growth   = (la_size - la_size_0) / la_size_0
    pv_remaining = pv_size / pv_size_0

    # ── EDT & component count ─────────────────────────────────────────
    la_uint = la_mask.astype(np.uint32, order='F')
    anisotropy=get_voxel_sizes(img)
    edt_la      = edt.edt(la_uint, anisotropy=anisotropy, black_border=True, parallel=-1)
    edt_in_pv   = edt_la * pv_mask.astype(np.float32)
    # Distance from each PV voxel to LA center of mass
    pv_indices = np.argwhere(pv_mask) 
    dist_to_com = np.linalg.norm(pv_indices - la_com, axis=1) 
    n_components, labeled_pv = count_pv_components(pv_mask)
    sizes = component_sizes(labeled_pv, n_components)
    largest_id = np.argmax(sizes) + 1  # +1 because label indices start at 1
    largest_mask = labeled_pv == largest_id
#     min_size = la_size_0 / 50  # or tune this

#     n_components, labeled_pv, all_sizes = count_large_components(
#     pv_mask, min_size
# )

    print("Number of components after pre-processing:", n_components)

#     # ── Compute balance among current components ──────────────────────
    balanced = is_balanced_separation(labeled_pv, n_components)
    print("Are componnets balanced:", balanced)


#     # Log every 10 iterations or when something changes
    if iteration % 10 == 0 or n_components != best_n_components:
        sizes = component_sizes(labeled_pv, n_components)
        print(f"Iter {iteration:4d} | LA growth: {la_growth*100:6.2f}% | "
                f"PV remaining: {pv_remaining*100:6.2f}% | "
                f"PV components: {n_components} | "
                f"sizes: {sizes} | balanced: {balanced}")

    if n_components > best_n_components:
        print(f"\n*** New PV component at iter {iteration}: "
                f"{best_n_components} -> {n_components} ***\n")
        best_n_components = n_components
        if separation_iteration is None:
            separation_iteration = iteration

#     # ── Stopping criteria ─────────────────────────────────────────────

#     # PRIMARY: target number of components AND they are size-balanced
    if n_components >= target_pv_components and balanced:
        print(f"Reached {target_pv_components} balanced PV components. Stopping.")
        break

    # SAFETY nets (unchanged logic, prevents infinite loop)
    if la_growth >= max_la_growth:
        print(f"LA growth limit reached ({la_growth*100:.1f}%). Stopping.")
        break
    if pv_remaining <= min_pv_remaining:
        print(f"PV too small ({pv_remaining*100:.1f}% remaining). Stopping.")
        break
    if pv_size == 0:
        print("PV mask is empty. Stopping.")
        break

    # ── Absorb the `step_size` PV voxels closest to LA boundary ──────
    # pv_indices    = np.argwhere(pv_mask)
    # pv_edt_values = edt_in_pv[pv_mask]
    # order         = np.argsort(pv_edt_values)

    #Including now both criteria with weights (CoM and Distance to boundry)
    #pv_indices    = np.argwhere(pv_mask)
    pv_indices = np.argwhere(largest_mask)
    pv_edt_values = edt_in_pv[largest_mask]
    # Distance to LA boundary
    #pv_edt_values = edt_in_pv[pv_mask]

    # Distance to center of mass (already computed above)
    dist_to_com = np.linalg.norm(pv_indices - la_com, axis=1)

    # Normalize both (important so scales don't dominate)
    pv_edt_norm = pv_edt_values / (pv_edt_values.max() + 1e-8)
    com_norm    = dist_to_com / (dist_to_com.max() + 1e-8)

    # Weights (tune these!)
    w_boundary = 0.3
    w_center   = 0.7

    # Combined score
    combined_score = w_boundary * pv_edt_norm + w_center * com_norm

    # Sort by combined score
    order = np.argsort(combined_score)
    n_to_assign   = min(step_size, len(order))

    for idx in pv_indices[order[:n_to_assign]]:
        seg[tuple(idx)] = la_label

# ── Final report ──────────────────────────────────────────────────────────
pv_final  = seg == pv_label
n_final, labeled_final = count_pv_components(pv_final)
sizes_final = component_sizes(labeled_final, n_final)
print(f"\nFinal state:")
print(f"  LA voxels     : {int((seg == la_label).sum())}")
print(f"  PV voxels     : {int(pv_final.sum())}")
print(f"  PV components : {n_final}  sizes: {sizes_final}")
if separation_iteration:
    print(f"  First split appeared at iteration: {separation_iteration}")



seg_result = relabel_pv_components(seg, pv_label,  affine=img.affine)

output= "7_separated_pvs_new.nii.gz"
save_nifti(output, seg_result, img)

print("File saved")