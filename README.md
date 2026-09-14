# Text-to-Image Generator using Stable Diffusion

This project is a simple **Text-to-Image Generator** made using Python, Stable Diffusion, Hugging Face Diffusers, PyTorch, and Gradio. The user enters a text prompt, and the application generates an image based on that prompt.

The project was developed mainly for learning and understanding the basic working of **Generative AI and diffusion models**. The application provides a simple interface where the user can select a model and device, enter a prompt, and change generation settings before creating an image.

## Features

* Generate images from text prompts
* Add a negative prompt to tell the model what to avoid
* Select between supported Stable Diffusion models
* Use GPU (CUDA), CPU, or automatic device selection
* Change image width and height
* Change inference steps and guidance scale
* Select different schedulers
* Use a seed for reproducible results
* View generation time and other generation details
* Save generated images with metadata
* View example prompts and recent generated images
* Includes a small learning section explaining diffusion and its main components

## Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Diffusers**
* **Stable Diffusion**
* **Gradio**
* **Google Colab**

## How It Works

1. The user enters a text prompt.
2. The prompt is passed to the Stable Diffusion pipeline.
3. The model processes the text and performs the diffusion-based image generation process.
4. The generated image is displayed through the Gradio interface.
5. The application also shows information such as image size, steps, guidance scale, scheduler, seed, device, and generation time.
6. When saving is enabled, the image and its metadata are stored locally.

## Dataset

**Dataset: Not applicable.**

This project does not use a separate project dataset. It uses pre-trained generative AI models for text-to-image generation.

## Example Prompt

```text
A realistic sea turtle sitting on a sandy beach, wearing black sunglasses and a red cap, ocean waves in the background, photorealistic, highly detailed
```

## Running the Project

The project was designed to run in **Google Colab** with GPU support.

Install the required packages and run the Python/Gradio code. After the interface starts, initialize a model, enter a prompt, and click **Generate Image**.

The application can create a public temporary Gradio link when `share=True` is used.

## Project Limitations

* Image generation is much slower without a suitable GPU.
* The model may not always follow every detail in a complex prompt.
* Higher resolution and more inference steps require more computing resources.
* The public Gradio link from Google Colab is temporary and depends on the running Colab session.

## Future Scope

The project can be improved by adding features such as image-to-image generation, inpainting, better prompt control, more advanced models, and a permanent GPU-based deployment.

## Project Type

**Generative AI / Text-to-Image / Diffusion Model Application**

## Author

**Zoya Kadodiya**



