from computer_vision_streamlit_app.plotting.inference_results import plot_segments_from_results
from computer_vision_streamlit_app.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder
from computer_vision_streamlit_app.plotting.color_distribution import plot_color_histogram
from computer_vision_streamlit_app.plotting.visualize_bin_mask import visualize_bin_mask
from computer_vision_streamlit_app.plotting.area_ratio_barplot import plot_grouped_ratio_barplot_with_labels, plot_single_segmented_ratio_barplot

from computer_vision_streamlit_app.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from computer_vision_streamlit_app.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask

from collections import defaultdict


folder_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/test/ver2/'#"/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/img/ver1/"
model_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/models/YOLOv8_seg/yolov8n_seg_w_aug.pt"

predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)
bin_masks = xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)

plot_segments_from_results(predictions[0][0])

# save prediction as pcikle
import pickle
with open('prediction.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Histogram
predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)
bin_masks = xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)
segment_color = SegmentColor(predictions[0][0].orig_img, bin_masks[0])
plot_color_histogram(segment_color.calculate_color_histogram())

# Bin mask
#visualize_bin_mask(bin_masks[2])

# Area ratio
segment_areas = defaultdict(list) # each cls can have multiple segments
labels = predictions[0][0].names
for cls, mask in zip(predictions[0][0].boxes.cls.tolist(), bin_masks):
    print("Process segment of class:", labels[cls])
    segment_area_obj = SegmentArea(predictions[0][0].orig_img, mask)
    print(f"{labels[cls]}, {segment_area_obj.calculate_area()}")
    segment_areas[labels[cls]].append(segment_area_obj)

area_ratios = segment_area_comparison(segment_areas)
area_ratios = area_ratios.iloc[0].to_dict()
plot_grouped_ratio_barplot_with_labels(area_ratios, list(predictions[0][0].names.values()))
print(predictions[0][0])

plot_single_segmented_ratio_barplot(area_ratios, list(predictions[0][0].names.values()))