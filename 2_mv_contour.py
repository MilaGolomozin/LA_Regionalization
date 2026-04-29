import argparse
import numpy as np
import nibabel as nib
import edt
from scipy import ndimage
import csv


# ── Default label configuration ───────────────────────────────────────────────
la_label = 2
lv_label = 3

#Loading the nifti file

img = nib.load("/7_separated_pvs_new.nii.gz")
seg_data = img.get_fdata().astype(np.int32)



def save_nifti(path, data, ref_img):
    out = nib.Nifti1Image(data.astype(np.int32), ref_img.affine, ref_img.header)
    nib.save(out, path)
    print(f"Saved: {path}")

def find_mitral_valve(seg, la_label, lv_label, img, mv_label=20, thickness_mm=4.0):
    """
    Detect the mitral valve as the narrow contact zone between LA and LV.

    Strategy:
      1. Compute EDT from the LA surface (distance of every non-LA voxel to LA).
      2. Compute EDT from the LV surface (distance of every non-LV voxel to LV).
      3. The mitral valve = voxels that are:
           - close to LA surface  (edt_la < threshold)
           - close to LV surface  (edt_lv < threshold)
           - NOT currently labelled as LA or LV themselves
           - in the gap/background between the two

    thickness_mm controls how thick the detected plane is on each side.
    """
    anisotropy = tuple(float(z) for z in img.header.get_zooms()[:3])

    la_mask = (seg == la_label).astype(np.uint32, order='F')
    lv_mask = (seg == lv_label).astype(np.uint32, order='F')

    # EDT gives distance FROM the surface of each structure
    # We invert: run EDT on the BINARY mask, which gives distance
    # to the nearest TRUE voxel from each FALSE voxel.
    # So edt(la_mask) = 0 inside LA, positive outside LA.
    # We want distance to the LA SURFACE from outside — 
    # that's just edt on the inverted mask, clamped to non-LA voxels.

    # Distance from LA boundary (measured outward into non-LA space)
    edt_from_la = edt.edt(
        (seg != la_label).astype(np.uint32, order='F'),
        anisotropy=anisotropy,
        black_border=False,
        parallel=-1
    )

    # Distance from LV boundary (measured outward into non-LV space)
    edt_from_lv = edt.edt(
        (seg != lv_label).astype(np.uint32, order='F'),
        anisotropy=anisotropy,
        black_border=False,
        parallel=-1
    )

    # Mitral valve candidate voxels:
    # - Not already labelled LA or LV (we're looking in the gap)
    # - Within thickness_mm of the LA surface
    # - Within thickness_mm of the LV surface
    not_la_or_lv = (seg != la_label) & (seg != lv_label)
    close_to_la  = edt_from_la < thickness_mm
    close_to_lv  = edt_from_lv < thickness_mm

    mv_mask = not_la_or_lv & close_to_la & close_to_lv

    n_voxels = int(mv_mask.sum())
    print(f"\nMitral valve detection:")
    print(f"  Threshold      : {thickness_mm} mm from each surface")
    print(f"  MV voxels found: {n_voxels}")

    if n_voxels == 0:
        print("  WARNING: No MV voxels found. Try increasing thickness_mm.")
        print("  Are LA and LV touching or very close in this segmentation?")
        return seg.copy()

    # Optionally: keep only the largest connected component
    # (removes stray voxels far from the main valve plane)
    labeled_mv, n_comp = ndimage.label(mv_mask)
    if n_comp > 1:
        sizes = [int((labeled_mv == i).sum()) for i in range(1, n_comp + 1)]
        largest = np.argmax(sizes) + 1
        mv_mask = labeled_mv == largest
        print(f"  Components found: {n_comp}, keeping largest ({sizes[largest-1]} voxels)")

    result = seg.copy()
    result[mv_mask] = mv_label

    print(f"  MV label assigned: {mv_label}")
    print(f"  Final MV voxels  : {int(mv_mask.sum())}")
    return result


# ── Usage ──────────────────────────────────────────────────────────────────
mv_label   = 20   # fixed label for mitral valve, consistent across scans

seg_with_mv = find_mitral_valve(
    seg       = seg_data,
    la_label  = la_label,
    lv_label  = lv_label,
    img       = img,
    mv_label  = mv_label,
    thickness_mm = 4.0    # tune this: 3–6 mm is a reasonable range
)

save_nifti("/7_with_mitral_valve_new.nii.gz", seg_with_mv, img)