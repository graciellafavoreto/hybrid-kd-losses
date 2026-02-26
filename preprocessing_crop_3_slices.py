import numpy as np
import pandas as pd
from pathlib import Path
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm
from scipy.ndimage import zoom

CROP_SIZE = 128   # size of origincal crop
# final size after bicubic upscaling (ImageNet and common CNN standards)
TARGET_SIZE = 224
NUM_SLICES = 3     # ck-1, ck, ck+1
CLEVEL = 0.5

output_dir = Path('preprocessing_crop_3slices128')
output_dir.mkdir(exist_ok=True)

print("Loading LIDC-IDRI scans...")
scans = pl.query(pl.Scan).all()
print(f"Total scans found: {len(scans)}")
thicknesses = [s.slice_thickness for s in scans]
print(
    f"Slice thickness — min: {min(thicknesses):.2f} | max: {max(thicknesses):.2f} | mean: {np.mean(thicknesses):.2f} mm")


def resize_bicubic(slice_2d, target=TARGET_SIZE):
    """Bicubic upscaling (order=3) from CROP_SIZE → TARGET_SIZE."""
    factor = target / slice_2d.shape[0]   # 224/128 = 1.75
    return zoom(slice_2d, factor, order=3)


all_metadata = []
erros = []

for scan in tqdm(scans, desc="Prcessing scans"):
    try:
        vol = scan.to_volume()
        nodule_clusters = scan.cluster_annotations()

        if len(nodule_clusters) == 0:
            continue

        patient_dir = output_dir / scan.patient_id
        patient_dir.mkdir(exist_ok=True)

        for nod_idx, cluster in enumerate(nodule_clusters):

            # 1 Consensus among annotators
            cmask, cbbox, _ = consensus(cluster, clevel=CLEVEL)
            ci = (cbbox[0].start + cbbox[0].stop) // 2
            cj = (cbbox[1].start + cbbox[1].stop) // 2
            ck = (cbbox[2].start + cbbox[2].stop) // 2

            # 2 XY crop 128×128 — sliding window centered on centroid
            half = CROP_SIZE // 2
            y0 = int(np.clip(ci - half, 0, vol.shape[0] - CROP_SIZE))
            y1 = y0 + CROP_SIZE
            x0 = int(np.clip(cj - half, 0, vol.shape[1] - CROP_SIZE))
            x1 = x0 + CROP_SIZE

            # 3 Z crop — 3 slices centered at ck
            z0 = int(np.clip(ck - 1, 0, vol.shape[2] - NUM_SLICES))
            z1 = z0 + NUM_SLICES

            crop = vol[y0:y1, x0:x1, z0:z1]  # (128, 128, 3)

            # 4 Bicubic upscaling 128 → 224 per slice
            slices_resized = [resize_bicubic(
                crop[:, :, s]) for s in range(NUM_SLICES)]
            slices_array = np.stack(slices_resized, axis=2)  # (224, 224, 3)

            if slices_array.shape != (TARGET_SIZE, TARGET_SIZE, NUM_SLICES):
                erros.append({'patient_id': scan.patient_id,
                              'error': f'Unexpected shape {slices_array.shape}'})
                continue

            # 5 Malignancy label
            malignancy_scores = [a.malignancy for a in cluster]
            mal_median = np.median(malignancy_scores)

            # 6 Save crop and metadata 
            np.save(patient_dir / f'nodule_{nod_idx}_crop.npy', slices_array)

            all_metadata.append({
                'patient_id': scan.patient_id,
                'nodule_id': nod_idx,
                'malignancy_median': mal_median,
                'malignancy_mean': np.mean(malignancy_scores),
                'num_annotations': len(cluster),
                'centroid_i': ci, 'centroid_j': cj, 'centroid_k': ck,
                'crop_y_min': y0,  'crop_y_max': y1,
                'crop_x_min': x0,  'crop_x_max': x1,
                'crop_z_min': z0,  'crop_z_max': z1,
            })

    except Exception as e:
        erros.append({'patient_id': scan.patient_id, 'error': str(e)})
        continue

df = pd.DataFrame(all_metadata)
df.to_csv(output_dir / 'nodules_metadata.csv', index=False)

print(f"\nProcessed nodules: {len(df)}")
print(f"Scans with errors: {len(erros)}")
if erros:
    for e in erros:
        print(f"  {e['patient_id']}: {e['error']}")
