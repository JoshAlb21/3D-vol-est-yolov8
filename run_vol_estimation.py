import os

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

import vol_est_yolov8 as vol_est
from vol_est_yolov8.plotting import inference_results
from vol_est_yolov8 import process_vol_est_main


def run_seg_inference(img_names:list, path_selected_model:str, task:str="segment"):

    model = YOLO(path_selected_model, task=task)
    predictions = []
    for image in img_names:

        # Raw image
        image = Image.open(image)

        # Run inference on the image
        prediction = model.predict(image)
        predictions.append(prediction[0])

        assert len(predictions) == 1, "Only one prediction object is expected for inference"

    return predictions



#predictions, img_names, first_segment_image = run_seg_inference(imgs_upload=imgs_upload, path_selected_model=path_selected_model_seg)

if __name__ == '__main__':

    model_path = "data/models/yolov8n-seg.pt"
    folder = "/Users/joshuaalbiez/Documents/python/3D-vol-est-yolov8/data/images"
    image_name = "test_image.png"
    num_orthogonal_lines = 150
    k_mm_per_px = 0.003118
    n_polynom_fallback = 3


    img_names = [os.path.join(folder, image_name)]

    predictions = run_seg_inference(img_names, model_path, task="segment")

    img_w_segmentation = inference_results.plot_segments_from_results(predictions[0], return_image=True)

    if predictions:
        # Volume estimation
        df_vol_res, first_res = process_vol_est_main.all_vol_est_main(img_names, predictions, k_mm_per_px, n_polynom_fallback, num_orthogonal_lines)

    #***************
    # Plot skeleton with orthogonal lines
    #***************
    drawer = vol_est.extract_skeleton.plot_skeleton.LineDrawer(points=first_res['fitted_points'], image=first_res['img_np'], orthogonal_lines=first_res['lines'], conv_2_rgb=False)
    img_w_center_line = drawer.get_img()
    plt.imshow(img_w_center_line)
    visualizer = vol_est.plotting.volume_visualizer.BodyVolumeVisualizer(first_res['lines'])
    volume_3d_fig = visualizer.visualize(return_fig=False)