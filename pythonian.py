import jetson_inference
import jetson_utils
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="resnet18", help="model to use, can be: googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--model", type=str, default="/home/nvidia/jetson-inference/python/training/classification/models/dogs/resnet18.onnx", help="model to use, can be: googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--labels", type=str, default="/home/nvidia/jetson-inference/python/training/classification/data/dogs/labels.txt", help="model to use, can be: googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--input_blob", type=str, default="input_0", help="name of the model's input layer")
parser.add_argument("--output_blob", type=str, default="output_0", help="name of the model's output layer")
# Change default output to PNG for better text quality
parser.add_argument("--output", type=str, default="output_image.png", help="filename to save the output image with classification")
opt = parser.parse_args()

# --- Define the test directory ---
TEST_DIR = "/home/nvidia/jetson-inference/python/training/classification/data/dogs/test"

# --- Choose a random image ---
def get_random_image_path(base_dir):
    breed_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not breed_dirs:
        raise FileNotFoundError(f"No breed directories found in {base_dir}")

    random_breed_dir = random.choice(breed_dirs)
    breed_path = os.path.join(base_dir, random_breed_dir)

    image_files = [f for f in os.listdir(breed_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {breed_path}")

    random_image_file = random.choice(image_files)
    return os.path.join(breed_path, random_image_file)

try:
    random_image_filename = get_random_image_path(TEST_DIR)
    print(f"Randomly selected image: {random_image_filename}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

img = jetson_utils.loadImage(random_image_filename)
net = jetson_inference.imageNet(model=opt.model, labels=opt.labels, input_blob=opt.input_blob, output_blob=opt.output_blob)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

output_text = f"{class_desc} (ID: {class_idx}) - {confidence*100:.2f}% Conf."

print(f"Image is recognized as {class_desc} (class #{class_idx}) with {confidence*100:.2f}% confidence")

# --- Drawing the text on the image with dynamic sizing ---
# Manually calculate font size based on image height.
# Increase the font_scale_factor to make the text proportionally larger.
# You might need to experiment with this value.
font_scale_factor = 15 # Was 25. A smaller divisor makes the text LARGER.
font_size = max(15, int(img.height / font_scale_factor)) # Increased minimum font size to 15
font = jetson_utils.cudaFont(size=font_size)

# Calculate the height of the text block to determine starting Y position
num_lines = output_text.count('\n') + 1
text_block_height = num_lines * font.GetSize()

# Choose a padding from the top and left
padding_x = 10
padding_y = 10

font.OverlayText(img, text=output_text, x=padding_x, y=padding_y, color=font.White, background=font.Gray50)

# --- Save the image with the overlay ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, opt.output)

# For JPEG, you could add a quality parameter if jetson_utils.saveImage supports it.
# E.g., jetson_utils.saveImage(output_path, img, quality=90) - check documentation.
# For now, changing to PNG is the easiest way to improve text clarity.
jetson_utils.saveImage(output_path, img)
print(f"Output image saved to {output_path}")

# Optional: If you want to display the image on a connected screen (requires a display)
# display = jetson_utils.glDisplay()
# while display.IsOpen():
#     display.RenderOnce(img)
#     display.SetTitle(f"Image Classification: {class_desc}")