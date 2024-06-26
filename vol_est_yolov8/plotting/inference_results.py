import cv2
import itertools
import torch
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    font_scale_inverse = 3
    lw = max(round(sum(image.shape) / font_scale_inverse * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    print("Label in box_label:", label)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image


def color_generator(fix_color_per_class:bool=False):
    # Define a list of colors
    # Use red, blue, greeen, yellow, orange, purple, brown, pink
    
    
    
    if fix_color_per_class:
        color_lookup = {'dog': (0, 255, 0), #green
                    'bicycle': (0, 0, 255), #blue
                    'umbrella': (255, 0, 0)} #red
        raise NotImplementedError("Fix color per class is not implemented yet.")
    else:
        base_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 165, 0), (128, 0, 128), (165, 42, 42), (255, 192, 203)]               
        color = itertools.cycle(base_colors)
    
    return color

def plot_mask(image, masks, single_mask_index=None, single_mask_array:np.ndarray=None):
    """
    Plot masks on the image.

    :param image: Original image as a numpy array.
    :param masks: List of masks, each mask is an array of normalized [x, y] points.
    :param single_mask_index: Index of the single mask to be plotted (optional).
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)

    img_height, img_width = image.shape[:2]

    if single_mask_index is not None:
        # Plot only the specified mask
        mask = masks[single_mask_index]
        pixel_coords = mask * np.array([img_width, img_height])
        polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)
    else:
        # Plot all masks
        if single_mask_array is not None:
            pixel_coords = single_mask_array * np.array([img_width, img_height])
            polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)
        else:
            for mask in masks:
                pixel_coords = mask * np.array([img_width, img_height])
                polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)

    plt.show()

def plot_segments(image, boxes, masks, cls:torch.tensor, labels:dict, conf:Union[torch.tensor, np.ndarray], score:bool=False, alpha=0.5, return_image:bool=False):
    
    color_iterable = color_generator()
    predefined_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    #cls = cls.numpy()
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("CLS", cls)
    print("LABELS", labels)

    # Note for my future me:
    #i is used to index boxes and masks, as they are ordered in the same sequence as the detections.
    #cls[i].item() is used to get the class ID for each detection, which is then used to fetch the correct label name and color.

    #for i, (box, mask_points_normalized) in enumerate(zip(boxes, masks)):
    # What was the error: we should not use the enumeration to select the boxes and masks, we should select the cls item!!!!!!!!
    #for i in range(len(boxes)):
    for i, class_id in enumerate(cls):
        #NEW- fix error
        #label_name = labels[cls[i].item()]
        label_name = labels[class_id.item()]
        print("Label_name", label_name)
        print("CLS i", class_id)
        if score:
            # THIS LINE IS FALSE!!
            #label = labels[cls[i]] + " " + str(round(100 * float(conf[i]), 1)) + "%"
            label = label_name + " " + str(round(100 * float(conf[i]), 1)) + "%"
        else:
            label = label_name

        #color = next(colors)
        #color = color_lookup[labels.get(cls[i].item())]
        color = next(color_iterable)
        
        
        # Draw bounding box
        #box_label(image, boxes[int(cls[i].item())], label, color)
        box_label(image, boxes[i], label, color)

        # Denormalize mask points
        #mask_points = masks[int(cls[i].item())].copy()
        mask_points = masks[i].copy()
        mask_points[:, 0] *= w
        mask_points[:, 1] *= h

        plot_mask(image, None, None, mask_points)

        # Convert denormalized mask points to a binary mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points.astype(np.int32)], 1)

        # Draw segmentation mask
        overlay = image.copy()
        for c in range(3):  # For each RGB channel
            overlay[..., c] = np.where(mask == 1, color[c], overlay[..., c])
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if return_image:
        return image
    else:
        cv2.imshow('Segments', image)
        cv2.waitKey(0)

def plot_segments_w_cogs2(image, boxes, masks, cls:torch.tensor, labels:Dict[int, str], cogs:Dict[str, Tuple[float, float]], alpha:0.5):
    colors = color_generator()
    cls = cls.numpy()
    h, w = image.shape[:2]

    # Manually drawing a line for testing
    line_color = (255, 105, 180)  # Pink in BGR
    start_point = (100, 100)  # You might need to adjust these coordinates
    end_point = (200, 200)    # You might need to adjust these coordinates
    cv2.line(image, start_point, end_point, line_color, 2)
    
    cv2.imshow('Segments', image)
    cv2.waitKey(0)


def plot_segments_w_cogs(image, boxes, masks, cls:torch.tensor, labels:Dict[int, str], cogs:Dict[str, Tuple[float, float]], alpha:0.5):
    colors = color_generator()
    cls = cls.numpy()
    h, w = image.shape[:2]
    image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # Define segment order and corresponding labels
    segment_order = ['dog', 'head_thorax', 'bicycle', 'thorax_abdomen', 'umbrella']

    # Define mapping between segment names and cogs dictionary keys
    segment_mapping = {'dog': '0', 'head_thorax': '0_1', 'bicycle': '1', 'thorax_abdomen': '1_2', 'umbrella': '2'}

    # Draw and connect CoGs based on segment order
    prev_cog = None
    line_color = (255, 105, 180)  # Define color for the lines (pink in BGR format)

    for segment in segment_order:
        # Use segment_mapping to get the correct key for cogs dictionary
        cog_key = segment_mapping.get(segment, None)
        cog = cogs.get(cog_key, None)

        if cog is not None:
            # Draw CoG with larger dot
            x, y = int(cog[0]), int(cog[1])
            cv2.circle(image, (x, y), 40, line_color, -1)

            # Draw line connecting CoGs with pink color and increased thickness
            if prev_cog is not None:
                cv2.line(image, (int(prev_cog[0]), int(prev_cog[1])), (x, y), line_color, 10)

            # Update previous CoG for next iteration
            prev_cog = cog

    # Draw bounding boxes and masks (if necessary)
    for i, (box, mask_points_normalized) in enumerate(zip(boxes, masks)):
        label = labels.get(cls[i], "")
        color = next(colors)

        # Draw bounding box
        box_label(image, box, label, color)
        # You may also draw masks here if necessary

    cv2.imshow('Segments', image)
    cv2.waitKey(0)

def order_cog_dict(cogs:dict, max_i:int) -> list:
    # Order cogs by '0', '1', ..., '2', ..., 'n'
    ordered_cogs = []
    for i in range(max_i):
        key = str(i)
        if key in cogs:
            ordered_cogs.append(cogs[key])

    return ordered_cogs

# Official ultralytics.engine.results.Results.plot() method
# Show the results
'''
for r in predictions[0]:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image

'''


def plot_segments_from_results(result, return_image:bool=False):
    """
    Plot bounding boxes and segmentation masks using a ultralytics.engine.results.Results object.

    Parameters:
    - result: An instance of ultralytics.engine.results.Results
    """
    # Extracting necessary attributes from the result object
    image = result.orig_img.copy()
    boxes = result.boxes.xyxy
    masks = result.masks.xyn
    cls = result.boxes.cls
    labels = result.names
    conf = result.boxes.conf
    score = True if conf is not None else False

    # Visualize bounding boxes and segmentations
    if return_image:
        image = plot_segments(image, boxes, masks, cls, labels, conf, score, return_image=return_image)
        print("Image type check inside inference_results.py")
        print(type(image))
        return image
    else:
        plot_segments(image, boxes, masks, cls, labels, conf, score, return_image=return_image)

def plot_binary_mask(mask):
    """
    Plot a binary mask using matplotlib.

    Parameters:
    - mask (np.ndarray): A 2D numpy array with binary values.
    """
    plt.imshow(mask, cmap='gray')
    plt.colorbar(label='Pixel Value')
    plt.title('Binary Mask Visualization')
    plt.xlabel('X-axis (Pixels)')
    plt.ylabel('Y-axis (Pixels)')
    plt.show()