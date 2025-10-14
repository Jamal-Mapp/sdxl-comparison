# # SDXL vs SDXL Turbo vs SD3.5 Comparison
A side-by-side comparison of SDXL, SDXL Turbo, and SD3.5 on identical prompts, analyzing generation speed, image quality, and FID scores with visual examples.

This project compares three Stable Diffusion models:

- **SDXL (Stable Diffusion XL)** – High-quality image generation
- **SDXL Turbo** – Optimized for speed
- **SD3.5 (Stable Diffusion 3.5)** – Previous generation model

We generate images from the same set of prompts and analyze:

1. **Generation speed**
2. **Visual quality**
3. **FID (Fréchet Inception Distance) score**

The goal is to highlight trade-offs between speed and quality across models.

---

## Average Generation Speed

| Model      | Avg. Time per Image (s) | Speedup vs SDXL |
|------------|------------------------|----------------|
| SDXL       | 50.12                  | 1x             |
| SDXL Turbo | 3.21                   | 15.6x          |
| SD3.5      | 48.97                  | 1.02x          |

---

## FID Score

The Fréchet Inception Distance (FID) measures visual quality similarity between SDXL (baseline) and other models.

---

## Usage

Follow these steps to reproduce the SDXL, SDXL Turbo, and SD3.5 comparison results:  

```bash
git clone https://github.com/yourusername/sdxl-comparison.git
cd sdxl-comparison
```

Install Dependencies  

Make sure you have Python 3.10+ installed, then install all required packages:  

```bash
pip install -r requirements.txt
```

Authenticate with Hugging Face  

This project requires access to the official Stability AI model weights.  
Log in to your Hugging Face account and generate a token.  
Open sdxl_comparison.py and replace:  

```python
hf_token = "YOUR_HUGGINGFACE_TOKEN"
```

Run the Comparison  

Execute the main script to generate and evaluate images:  

```bash
python sdxl_comparison.py
```

The script will:  
- Generate images for each model using the same prompts.  
- Display side-by-side comparisons.  
- Compute the FID (Fréchet Inception Distance) for quality evaluation.  
- Print average generation times and speedup ratios.

View Results  

Images will be saved in:  
sdxl_images/  
sdxl_turbo_images/  
sd3_5_images/  
