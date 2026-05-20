import numpy as np

from ssm_extract_ufe_viz.image_metrics import (
    activation_map_iou,
    lpips_distance,
    ssim_distance,
    topk_overlap,
)


def test_image_distances_identical_images_are_zero():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[4:8, 4:8] = 1.0
    assert ssim_distance(image, image) < 1e-6
    assert lpips_distance(image, image) < 1e-6


def test_activation_map_iou_and_topk_overlap():
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.zeros((8, 8), dtype=np.float32)
    a[:4, :4] = 1.0
    b[:4, :4] = 1.0
    assert activation_map_iou(a, b) == 1.0
    assert topk_overlap([1, 2, 3], [3, 4, 5]) == 1 / 3
