import napari
import numpy as np

from scripts.segmentation.config import NUM_CLASSES, CLASS_NAMES, CLASS_COLORS


def add_image_layers(viewer, sparse_display, dense_volume):

    viewer.add_image(sparse_display, name="Sparse Input", colormap="gray", opacity=1.0, rendering="attenuated_mip")
    viewer.add_image(dense_volume, name="Reconstructed Volume", colormap="gray", opacity=1.0, rendering="attenuated_mip")

def add_segmentation_layers(viewer, class_vols):

    for c in range(NUM_CLASSES):
        rgba = CLASS_COLORS[c].astype(float) / 255.0
        color_dict = {0: np.array([0, 0, 0, 0]), c + 1: rgba}
        layer = viewer.add_labels(class_vols[c], name=f"Seg {c}: {CLASS_NAMES[c]}", opacity=0.6, iso_gradient_mode='smooth')
        layer.color = color_dict

def launch_viewer(sparse_display, dense_volume, class_vols):

    viewer = napari.Viewer(title="OCT Segmented Volume")
    add_image_layers(viewer, sparse_display, dense_volume)
    add_segmentation_layers(viewer, class_vols)
    napari.run()
