# =============================
# SDXL vs SDXL Turbo vs SD3.5 Comparison
# Student 17 (Prompts 161–170)
# =============================

# -----------------------------
# Hugging Face Authentication
# -----------------------------
from huggingface_hub import login
hf_token = "YOUR_HUGGINGFACE_TOKEN"
login(hf_token)

# -----------------------------
# Imports
# -----------------------------
import torch, time, os, gc
from diffusers import DiffusionPipeline
from datasets import load_dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load prompts 161–170
# -----------------------------
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test")
start, end = 160, 170  # zero-based
my_prompts = dataset[start:end]['Prompt']

# -----------------------------
# Folders to save images
# -----------------------------
os.makedirs("sdxl_images", exist_ok=True)
os.makedirs("sdxl_turbo_images", exist_ok=True)
os.makedirs("sd3_5_images", exist_ok=True)

# -----------------------------
# Function to generate images
# -----------------------------
def generate_images(prompts, model_name, output_folder, width=512, height=512, steps=50):
    pipe = DiffusionPipeline.from_pretrained(model_name, dtype=torch.float16).to(device)
    times = []
    for i, prompt in enumerate(prompts, start=161):
        start_time = time.time()
        if "turbo" in model_name:
            guidance_scale = 0.0
            num_steps = 1
        elif "sdxl" in model_name:
            guidance_scale = 7.5
            num_steps = steps
        else:
            # SD3.5
            guidance_scale = 7.0
            num_steps = steps
        img = pipe(prompt=prompt, width=width, height=height, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]
        elapsed = time.time() - start_time
        times.append(elapsed)
        img.save(f"{output_folder}/img_{i}.png")
        print(f"Saved {output_folder}/img_{i}.png ({elapsed:.2f}s)")
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return times

# -----------------------------
# Generate SDXL images
# -----------------------------
print("\nGenerating SDXL images...")
sdxl_times = generate_images(my_prompts, "stabilityai/stable-diffusion-xl-base-1.0", "sdxl_images", steps=50)

# -----------------------------
# Generate SDXL Turbo images
# -----------------------------
print("\nGenerating SDXL Turbo images...")
turbo_times = generate_images(my_prompts, "stabilityai/sdxl-turbo", "sdxl_turbo_images", steps=1)

# -----------------------------
# Clear Cache
# -----------------------------
print("\nClearing cache before SD3.5 generation...")
torch.cuda.empty_cache()
gc.collect()

# -----------------------------
# Generate SD3.5 images
# -----------------------------
print("\nGenerating SD3.5 images...")
sd35_times = generate_images(my_prompts, "stabilityai/stable-diffusion-3.5-large", "sd3_5_images", steps=50)

# -----------------------------
# Display sample comparisons (SDXL vs Turbo vs SD3.5)
# -----------------------------
print("\nDisplaying side-by-side comparisons...")
for i in range(161, 171):
    img1 = Image.open(f"sdxl_images/img_{i}.png")
    img2 = Image.open(f"sdxl_turbo_images/img_{i}.png")
    img3 = Image.open(f"sd3_5_images/img_{i}.png")
    prompt = my_prompts[i-161]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1)
    axes[0].set_title(f"SDXL\n{sdxl_times[i-161]:.2f}s")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(f"SDXL Turbo\n{turbo_times[i-161]:.2f}s")
    axes[1].axis("off")

    axes[2].imshow(img3)
    axes[2].set_title(f"SD3.5\n{sd35_times[i-161]:.2f}s")
    axes[2].axis("off")

    plt.suptitle(prompt, fontsize=10, wrap=True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Compute FID (SDXL vs Turbo vs SD3.5)
# -----------------------------
print("\nComputing FID (Fréchet Inception Distance)... this may take a few minutes.")

fid = FrechetInceptionDistance().to(device)
transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor()
])

for i in range(161, 171):
    img1 = Image.open(f"sdxl_images/img_{i}.png").convert("RGB")
    img2 = Image.open(f"sdxl_turbo_images/img_{i}.png").convert("RGB")
    img3 = Image.open(f"sd3_5_images/img_{i}.png").convert("RGB")

    img1_tensor = (transform(img1) * 255).to(torch.uint8)
    img2_tensor = (transform(img2) * 255).to(torch.uint8)
    img3_tensor = (transform(img3) * 255).to(torch.uint8)

    # SDXL = real baseline, Turbo + SD3.5 = generated variants
    fid.update(img1_tensor.unsqueeze(0).to(device), real=True)
    fid.update(img2_tensor.unsqueeze(0).to(device), real=False)
    fid.update(img3_tensor.unsqueeze(0).to(device), real=False)

fid_score = fid.compute().item()
print(f"\nFID Score (SDXL vs SDXL Turbo vs SD3.5): {fid_score:.4f}")

# -----------------------------
# Compare Average Speeds
# -----------------------------
avg_sdxl_time = sum(sdxl_times) / len(sdxl_times)
avg_turbo_time = sum(turbo_times) / len(turbo_times)
avg_sd35_time = sum(sd35_times) / len(sd35_times)

print(f"\nAverage SDXL Time per Image: {avg_sdxl_time:.2f} seconds")
print(f"Average SDXL Turbo Time per Image: {avg_turbo_time:.2f} seconds")
print(f"Average SD3.5 Time per Image: {avg_sd35_time:.2f} seconds")

speedup_turbo = avg_sdxl_time / avg_turbo_time
speedup_sd35 = avg_sdxl_time / avg_sd35_time

print(f"\nSpeedup Ratio (SDXL / SDXL Turbo): {speedup_turbo:.2f}x faster")
print(f"Speedup Ratio (SDXL / SD3.5): {speedup_sd35:.2f}x faster")