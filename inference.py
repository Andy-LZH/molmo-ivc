from PartImageNet.spin import SPIN_Dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import re
import numpy as np
from typing import List, Tuple
import json
import os
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader
from tqdm import tqdm

def is_valid_format(input_string):
    # Define the regular expression pattern
    pattern = re.compile(
        r"^(\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)\n?)+$|"
        r'^<point\s+x="\s*\d+(\.\d+)?"\s+y="\s*\d+(\.\d+)?"\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(x\d+="\s*\d+(\.\d+)?"\s+y\d+="\s*\d+(\.\d+)?"\s+)+alt="[\s\S]*?">[\s\S]*?</points>$|'
        r'^<point\s+p=\s*\d{3}\s*,\s*\d{3}\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(\d+=\s*\d{3}\s*,\s*\d{3}\s+)+alt="[\s\S]*?">[\s\S]*?</points>$'
    )

    # Match the entire input string against the pattern
    match = pattern.fullmatch(input_string.strip())

    # Return True if the match is successful, False otherwise
    return match is not None


# NOTE: this function calculate the precision and recall of the points
def is_point_in_region(point: Tuple[float, float], mask: np.ndarray) -> bool:
    """
    Check if the point (x, y) is within the region defined by the boolean mask.

    Parameters:
    - point (tuple of floats): x/y-coordinate of the point
    - mask (2D numpy array): Boolean mask of shape [H, W] representing the region

    Returns:
    - bool: True if the point is within the region, False otherwise
    """
    height, width = mask.shape
    x, y = point

    # Round the coordinates to the nearest integer
    x_int = int(round(x))
    y_int = int(round(y))

    # Check if the rounded point is within the bounds of the image
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False

    # Check if the point is within the region
    return mask[y_int, x_int]


def compute_precision(
    row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]
):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(preds)


def compute_recall(
    row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]
):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(masks)


def f1_score(precision: float, recall: float, epsilon: float = 1e-10):
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + epsilon)


def extract_points(text, image_w, image_h):
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text
    ):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r"(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})", text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points


annotation_dir = "PartImageNet/jsons"
image_dir = "PartImageNet/images"
split = "test"
granularity = "part"

SPIN = SPIN_Dataset(
    granularity=granularity,
    annotation_dir=annotation_dir,
    image_dir=image_dir,
    split=split,
)

image, annotations = SPIN[0]
print(image)
print(annotations)
#

# load the processor
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

print("Model loaded")

# Training loop starts here TODO: make this as torch dataloader and calculate precision and recall for each image and save the pointing results, and then
# Adjust the DataLoader for a batch size of 1
# dataloader = DataLoader(SPIN, batch_size=1, shuffle=True, num_workers=4)

test_data = []

for idx, (image, annotations) in tqdm(enumerate(SPIN), total=len(SPIN)):
    ## process the image and text
    inputs = processor.process(
        images=[
            image,
        ],
        text=annotations["prompts"],
    )

    ## move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    ## generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    ## only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if not is_valid_format(generated_text):
        print(f"Invalid format: {generated_text}")
        results = {
            "generated_text": generated_text,
            "generated_points": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
        test_data.append(results)
        # continue to the next image
        print("Skipping to the next image")
        continue

    # convert generated task points to a list of tuples
    generated_points = extract_points(generated_text, image.size[0], image.size[1])
    generated_points = np.array(generated_points)
    print("Generated points:", generated_points)
    binary_mask = maskUtils.decode(annotations["segmentation"])

    # # calculate precision and recall
    # precision = compute_precision(
    #     row_ind=np.arange(len(generated_points)),
    #     col_ind=np.zeros(len(generated_points)),
    #     preds=generated_points,
    #     masks=[binary_mask],
    # )
    # recall = compute_recall(
    #     row_ind=np.arange(len(generated_points)),
    #     col_ind=np.zeros(len(generated_points)),
    #     preds=generated_points,
    #     masks=[binary_mask],
    # )
    # f1 = f1_score(precision, recall)
    # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    # save the results

    # save visulizationy with original image, mask overlayed on the image, and generated points on same image
    plt.imshow(image)
    plt.imshow(binary_mask, alpha=0.5)
    plt.scatter(
        generated_points[:, 0], generated_points[:, 1], c="red", s=10, label="Generated Points"
    )
    plt.axis("off")
    plt.savefig(f"PartImageNet/Molmo7B/generated_points_{len(test_data)}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    results = {
        "generated_text": generated_text, 
        "generated_points": generated_points.tolist(),
        # "precision": precision,
        # "recall": recall,
        # "f1_score": f1,
        "output_image": f"PartImageNet/Molmo7B/generated_points_{len(test_data)}.png",
    }
    annotations.update(results)
    test_data.append(annotations)
    break


# save the results to a json file
with open("test_results.json", "w") as f:
    json.dump(test_data, f, indent=4)
print("Results saved to test_results.json")