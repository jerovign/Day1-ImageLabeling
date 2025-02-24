
# Note: don't enforce certain Python versions in reality; 
# this is just to help learn how to install different Pyhton versions when needed.
import sys
if sys.version_info.major != 3 or sys.version_info.minor != 11:
    raise RuntimeError("This script should be run by Python 3.11.")


import napari
import numpy as np
from skimage.data import coins
from skimage.segmentation import flood
from skimage.measure import label, regionprops
from skimage.draw import disk

# Global set to track which seed points have been processed.
processed_points = set()

def detect_coin_disk(image, point, tolerance=200):
    """
    Given a grayscale image and a seed point (row, col), this function:
      1. Uses region-growing (flood) segmentation with the given tolerance to capture
         the coin’s connected region.
      2. Labels the region and identifies the region that contains the seed point.
      3. Computes the region’s centroid and equivalent diameter.
      4. Creates a disk mask (via skimage.draw.disk) centered at the computed centroid,
         with a radius of half the equivalent diameter.
    
    Returns a boolean mask (same shape as the image) where the fitted disk is True.
    If segmentation fails or no region is found, returns None.
    """
    y, x = int(round(point[0])), int(round(point[1]))
    
    # Use flood segmentation to capture the coin region around the seed point.
    try:
        coin_mask = flood(image, (y, x), tolerance=tolerance)
    except Exception:
        return None
    if not coin_mask.any():
        return None
    
    # Label the segmented region.
    labeled = label(coin_mask)
    
    # Identify the region that contains the seed.
    selected_region = None
    for region in regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        if minr <= y < maxr and minc <= x < maxc:
            selected_region = region
            break
    if selected_region is None:
        return None

    # Use the region’s centroid and equivalent diameter to define the disk.
    centroid = selected_region.centroid  # (y, x) in float coordinates
    eq_diam = selected_region.equivalent_diameter
    radius = int(round(eq_diam / 2))
    
    # Create a disk mask on a full-image canvas.
    disk_mask = np.zeros_like(image, dtype=bool)
    rr, cc = disk((int(round(centroid[0])), int(round(centroid[1]))), radius, shape=image.shape)
    disk_mask[rr, cc] = True
    return disk_mask

def on_points_change(event):
    global processed_points
    # Loop over all points in the points layer.
    for i, point in enumerate(points_layer.data):
        if i in processed_points:
            continue
        # Compute the best-fit disk for the coin region around the seed.
        disk_mask = detect_coin_disk(image_data, point, tolerance=30)
        if disk_mask is not None:
            # Label the fitted disk in the labels layer (label index = seed index + 1).
            labels_data[disk_mask] = i + 1
            labels_layer.data = labels_data  # update display
        processed_points.add(i)

# Load the coins image.
image_data = coins()

# Create a napari viewer.
viewer = napari.Viewer()

# Add the coins image layer.
image_layer = viewer.add_image(image_data, name='coins')

# Create an empty labels layer (same shape as image, integer type).
labels_data = np.zeros_like(image_data, dtype=int)
labels_layer = viewer.add_labels(labels_data, name='labels')

# Add an empty points layer.
points_layer = viewer.add_points(np.empty((0, 2)), name='points')

# Connect the points layer event to our callback.
points_layer.events.data.connect(on_points_change)

# Start the napari event loop.
napari.run()
