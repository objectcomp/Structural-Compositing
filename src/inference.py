import sys
if './' not in sys.path:
    sys.path.append('./')

import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
import matplotlib.pyplot as plt
import os

# Load the model configuration and checkpoint
model = create_model('./configs/comp_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpt/comp_insect.ckpt', location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(
    canny_image=None,  
    hed_image=None,
    midas_image=None,
    prompt='',
    a_prompt='',
    n_prompt='',
    num_samples=1,
    image_resolution=512,
    ddim_steps=50,
    strength=1.0,
    scale=7.5,
    seed=42,
    eta=0.0,
    global_strength=1.0,
    output_dir='./output_images',
    object_scale=1.0,
    object_pos_x=0.5,
    object_pos_y=0.5
):
    """
    Generates images using canny, hed, and midas as local control signals.
    The user can specify a scaling factor and position for the foreground object
    in the Canny/HED image, determined by bounding-box logic.
    """

    seed_everything(seed)

    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the anchor image for resizing
    anchor_image = None
    for img in [canny_image, hed_image, midas_image]:
        if img is not None:
            anchor_image = img
            break
    if anchor_image is None:
        raise ValueError("At least one local control image (canny, hed or midas) must be provided.")

    # Resize images to match the desired resolution
    H, W = image_resolution, image_resolution
    anchor_image = cv2.resize(anchor_image, (W, H))

    # Prepare empty control map with 9 channels (3 images × 3 channels each)
    num_channels = 9  # canny + hed + midas
    local_control = np.zeros((H, W, num_channels), dtype=np.float32)

    # Mapping for channels
    control_channel_mapping = {
        'canny': (0, 3),
        'hed':   (3, 6),
        'midas': (6, 9),
    }

    def process_control_image(control_image, start_channel, end_channel, scale=1.0, pos_x=0.5, pos_y=0.5):
        """
        Resizes the control_image to (W,H), then if scale !=1.0 or pos_x/pos_y !=0.5,
        finds the largest contour (the foreground object), scales it, and repositions it.
        Returns the bounding box if a contour is found.
        """
        # Resize the control image
        control_image = cv2.resize(control_image, (W, H))

        bbox = None  # Initialize bounding box as None

        # Scale/reposition the object if requested
        if scale != 1.0 or pos_x != 0.5 or pos_y != 0.5:
            # Convert to grayscale if needed
            if control_image.ndim == 3 and control_image.shape[2] == 3:
                control_gray = cv2.cvtColor(control_image, cv2.COLOR_BGR2GRAY)
            else:
                control_gray = control_image

            # Find contours of the object
            contours, _ = cv2.findContours(control_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Create a mask of the object
                mask = np.zeros_like(control_gray)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

                # Get bounding rectangle of the object
                x, y, w, h = cv2.boundingRect(mask)

                # Extract the object region
                object_region = control_image[y:y+h, x:x+w]

                # Calculate new size
                new_w = int(w * scale)
                new_h = int(h * scale)

                # Resize the object region
                object_region_scaled = cv2.resize(object_region, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Create a new blank canvas
                scaled_control_image = np.zeros_like(control_image)

                # Calculate new position based on pos_x and pos_y
                max_pos_x = W - new_w
                max_pos_y = H - new_h

                # Ensure positions are within [0, 1]
                pos_x = np.clip(pos_x, 0.0, 1.0)
                pos_y = np.clip(pos_y, 0.0, 1.0)

                # Final position
                obj_x = int(pos_x * max_pos_x)
                obj_y = int(pos_y * max_pos_y)

                # Place the scaled object onto the canvas
                scaled_control_image[obj_y:obj_y+new_h, obj_x:obj_x+new_w] = object_region_scaled

                control_image = scaled_control_image

                # Save bounding box coordinates
                bbox = (obj_x, obj_y, obj_x + new_w, obj_y + new_h)
            else:
                print("No contours found in the control image.")
                # If no contour is found, we keep the original image

        # Convert grayscale images to RGB if necessary
        if control_image.ndim == 2 or (control_image.shape[2] == 1):
            control_image = cv2.cvtColor(control_image, cv2.COLOR_GRAY2RGB)

        # Normalize to [0, 1]
        control_image = control_image.astype(np.float32) / 255.0
        local_control[:, :, start_channel:end_channel] = control_image

        return bbox  # Return bounding box

    # We can keep track of only the last bounding box found if needed
    bbox = None

    # Process each control image (canny, hed, midas) if provided
    if canny_image is not None:
        bbox = process_control_image(
            canny_image,
            *control_channel_mapping['canny'],
            scale=object_scale,
            pos_x=object_pos_x,
            pos_y=object_pos_y
        )
    if hed_image is not None:
        bbox = process_control_image(
            hed_image,
            *control_channel_mapping['hed'],
            scale=object_scale,
            pos_x=object_pos_x,
            pos_y=object_pos_y
        )
    if midas_image is not None:
        # Typically midas is a depth map, so we skip bounding-box logic
        process_control_image(
            midas_image,
            *control_channel_mapping['midas'],
            scale=1.0,
            pos_x=0.5,
            pos_y=0.5
        )

    # Convert local_control to tensor and prepare batch
    local_control_tensor = torch.from_numpy(local_control).float().cuda()
    local_control_tensor = local_control_tensor.permute(2, 0, 1)  # (C, H, W)
    local_control_tensor = local_control_tensor.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # (B, C, H, W)

    # Prepare global control (content embedding)
    global_control_tensor = torch.zeros((num_samples, 768)).float().cuda()

    # Prepare conditioning
    cond = {
        "local_control": [local_control_tensor],
        "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
        "global_control": [global_control_tensor],
    }
    un_cond = {
        "local_control": [local_control_tensor],  # Same local control for unconditional
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        "global_control": [torch.zeros_like(global_control_tensor)],
    }
    shape = (4, H // 8, W // 8)

    # Set control scales (13 typical for multi-scale)
    model.control_scales = [strength] * 13

    # Generate samples
    samples, _ = ddim_sampler.sample(
        ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond, global_strength=global_strength
    )

    # Decode samples to images
    x_samples = model.decode_first_stage(samples)
    x_samples = (x_samples * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    x_samples = x_samples.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

    results = [x_samples[i] for i in range(num_samples)]

    # Save the generated images
    for idx, img in enumerate(results):
        # Convert the decoded image (RGB) to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Save the basic generated image
        filename = os.path.join(output_dir, f"generated_image_{idx+1}.png")
        cv2.imwrite(filename, img_bgr)
        print(f"Image saved at: {filename}")

        if bbox is not None:
            # Unpack bounding box coordinates
            x_min, y_min, x_max, y_max = bbox

            # Draw rectangle on the BGR image
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # Save the image with bounding box
            filename_bbox = os.path.join(output_dir, f"generated_image_{idx+1}_bbox.png")
            cv2.imwrite(filename_bbox, img_bgr)
            print(f"Image with bounding box saved at: {filename_bbox}")

            # Save bounding box coordinates in a text file
            bbox_filename = os.path.join(output_dir, "bounding_boxes.txt")
            with open(bbox_filename, 'a') as f:
                f.write(f"{filename},{x_min},{y_min},{x_max},{y_max}\n")

            # Display with bounding box (optional)
            img_bbox_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 8))
            plt.imshow(img_bbox_rgb)
            plt.title(f"Generated Image {idx+1} with BBox")
            plt.axis('off')
            plt.show()

    return results


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Set paths to your control images
    canny_image_path = ''     # Replace with your Canny edge image path
    hed_image_path   = ''         # Replace with your HED edge image path
    midas_image_path = ''# Replace with your MiDaS depth map path

    # Load control images (set to None if not used)
    canny_image = cv2.imread(canny_image_path) if canny_image_path else None
    hed_image   = cv2.imread(hed_image_path)   if hed_image_path else None
    midas_image = cv2.imread(midas_image_path) if midas_image_path else None

    # Verify that at least one control image is provided
    if all(img is None for img in [canny_image, hed_image, midas_image]):
        raise ValueError("At least one local control image must be provided.")

    # Set your prompts
    # (Changed “insect” -> “object” in the prompt)
    prompt = "   "
    a_prompt = "best quality, HD Quality, 8K"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, low quality"

    # Set parameters
    num_samples = 4  # Number of images to generate
    image_resolution = 512
    ddim_steps = 50
    strength = 2.0
    scale = 10.0
    seed = 7070
    eta = 0.0
    global_strength = 0.0
    output_dir = './output_images'

    # Set the object scaling factor and position
    object_scale = 0.45   # Adjust object size
    object_pos_x = 0.5    # Horizontal position (0.0 to 1.0)
    object_pos_y = 0.5    # Vertical position   (0.0 to 1.0)

    # Run the process function
    results = process(
        canny_image=canny_image,
        hed_image=hed_image,
        midas_image=midas_image,
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        ddim_steps=ddim_steps,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta,
        global_strength=global_strength,
        output_dir=output_dir,
        object_scale=object_scale,     # Pass the scaling factor
        object_pos_x=object_pos_x,     # Pass the horizontal position
        object_pos_y=object_pos_y      # Pass the vertical position
    )

    # Display the results
    for i, img in enumerate(results):
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Generated Image {i+1}")
        plt.axis('off')
        plt.show()
