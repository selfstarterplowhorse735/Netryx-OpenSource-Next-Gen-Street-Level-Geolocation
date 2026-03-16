"""
Standalone index builder — Ultra-Fast Chunked Version.
Fixes OOM Crash (137) and Slowness by:
1. np.memmap for disk-backed array
2. Batch-processing part files (instead of single-entry loop)
3. Normalized in-place in batches before writing to high-speed memmap

Usage:
  1. Close test_super.py
  2. Run: python3 build_index.py
"""
import numpy as np
import os
import glob
import gc
import time

# check if EXPANSION disk exists, otherwise use local folder
_potential_dir = "/Volumes/Expansion/netryx"
if os.path.exists(_potential_dir):
    DATA_DIR = _potential_dir
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netryx_data")

COSPLACE_PARTS_DIR = os.path.join(DATA_DIR, "cosplace_parts")
EMB_CSV = os.path.join(DATA_DIR, "embeddings_index.csv")
COMPACT_INDEX_DIR = os.path.join(DATA_DIR, "index")
COMPACT_DESCS_PATH = os.path.join(COMPACT_INDEX_DIR, "cosplace_descriptors.npy")
COMPACT_META_PATH = os.path.join(COMPACT_INDEX_DIR, "metadata.npz")
MEMMAP_TEMP_PATH = os.path.join(COMPACT_INDEX_DIR, "build_temp.mmap")

os.makedirs(COMPACT_INDEX_DIR, exist_ok=True)

def parse_emb_path(emb_path):
    filename = os.path.basename(emb_path)
    parts = filename.replace('.npz', '').rsplit('_', 1)
    if len(parts) == 2:
        try: return parts[0], int(parts[1])
        except ValueError: pass
    return None, None

print("=== STEP 1/5: Loading Coordinate Map (CSV) ===")
csv_locations = {}
if os.path.exists(EMB_CSV):
    with open(EMB_CSV, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3: csv_locations[os.path.basename(parts[0])] = (float(parts[1]), float(parts[2]))
print(f"  Loaded {len(csv_locations)} coords.")

print("=== STEP 2/5: Identifying valid entries ===")
pattern = os.path.join(COSPLACE_PARTS_DIR, "cosplace_part_*.npz")
part_files = sorted(glob.glob(pattern))
dim = None
total_count = 0
valid_by_file = [] # List of (pf, valid_mask, count)

for pf in part_files:
    data = np.load(pf, allow_pickle=True)
    if dim is None: dim = data['descriptors'].shape[1]
    
    paths = data['paths']
    has_embedded = 'lats' in data and 'lons' in data
    
    mask = np.zeros(len(paths), dtype=bool)
    for j, path in enumerate(paths):
        basename = os.path.basename(str(path))
        if has_embedded and (float(data['lats'][j]) != 0 or float(data['lons'][j]) != 0):
            mask[j] = True
        elif basename in csv_locations:
            mask[j] = True
            
    count = np.sum(mask)
    valid_by_file.append((pf, mask, count))
    total_count += count
    del data
    if len(valid_by_file) % 200 == 0: print(f"  Scanned {len(valid_by_file)}/{len(part_files)} files...")

print(f"Total entries: {total_count} / Dim: {dim}")

print(f"=== STEP 3/5: Allocating {total_count * dim * 4 / 1024**2:.1f} MB Memmap ===")
if os.path.exists(MEMMAP_TEMP_PATH): os.remove(MEMMAP_TEMP_PATH)
all_descs = np.memmap(MEMMAP_TEMP_PATH, dtype='float32', mode='w+', shape=(total_count, dim))

print("=== STEP 4/5: Batch Processing & Normalizing ===")
lats = np.zeros(total_count, dtype=np.float32)
lons = np.zeros(total_count, dtype=np.float32)
headings = np.zeros(total_count, dtype=np.int16)
panoids = []
final_paths = []
write_idx = 0
t0 = time.time()

for i, (pf, mask, count) in enumerate(valid_by_file):
    if count == 0: continue
    
    data = np.load(pf, allow_pickle=True)
    
    # Batch grab descriptors
    descs = data['descriptors'][mask].astype(np.float32)
    # Batch normalize
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    descs /= norms
    
    # Batch write to memmap
    all_descs[write_idx : write_idx + count] = descs
    
    # Metadata extraction
    paths_subset = data['paths'][mask]
    has_embedded = 'lats' in data and 'lons' in data
    
    lats_subset = np.zeros(count, dtype=np.float32)
    lons_subset = np.zeros(count, dtype=np.float32)
    
    for j in range(count):
        path = str(paths_subset[j])
        p, h = parse_emb_path(path)
        panoids.append(p or "")
        headings[write_idx + j] = h or 0
        final_paths.append(path)
        
        # Priority: Embedded > CSV
        if has_embedded and (float(data['lats'][mask][j]) != 0):
            lats_subset[j] = float(data['lats'][mask][j])
            lons_subset[j] = float(data['lons'][mask][j])
        else:
            lat, lon = csv_locations[os.path.basename(path)]
            lats_subset[j] = lat
            lons_subset[j] = lon
            
    lats[write_idx : write_idx + count] = lats_subset
    lons[write_idx : write_idx + count] = lons_subset
    
    write_idx += count
    del data, descs, norms
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(part_files)} files ({write_idx}/{total_count} entries) [{time.time()-t0:.1f}s]")

all_descs.flush()

print("=== STEP 5/5: Finalizing Files ===")
np.save(COMPACT_DESCS_PATH, all_descs)
np.savez_compressed(COMPACT_META_PATH,
    lats=lats, lons=lons, headings=headings,
    panoids=np.array(panoids, dtype=object),
    paths=np.array(final_paths, dtype=object)
)

if os.path.exists(MEMMAP_TEMP_PATH): os.remove(MEMMAP_TEMP_PATH)
print(f"\n✅ SUCCESS: Built index with {total_count:,} entries in {time.time()-t0:.1f}s")
