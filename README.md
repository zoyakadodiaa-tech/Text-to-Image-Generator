# Text-to-Image Generation with Stable Diffusion, LoRA, CLIP, and GANs

**Author:** Zoya Kadodiya\
**Primary notebook:** `Text_to_Image_Merged (1).ipynb`\
**Project type:** Generative AI · Text-to-Image · Diffusion Models ·
GANs

## Project Overview

This repository extends an existing **Stable Diffusion text-to-image
generator** with experiments in flower-domain generation. The original
application uses Python, PyTorch, Hugging Face Diffusers, Stable
Diffusion, and Gradio to generate images from user prompts through an
interactive interface.

The additional work documented in `Text_to_Image_Merged (1).ipynb`
explores six tasks using the existing pretrained Stable Diffusion model
as the foundation for the diffusion/LoRA work, alongside separate
GAN-based experiments. These tasks cover LoRA fine-tuning,
class-conditional GANs, CLIP text embeddings, flower dataset analysis,
attention-based GANs, and an integrated prompt-to-image workflow.

The diffusion and GAN approaches serve different purposes: Stable
Diffusion is a pretrained latent diffusion model, while the GAN
experiments train separate generators and discriminators. The GANs are
not components of the Stable Diffusion architecture.

## Existing Stable Diffusion Application

The base application allows users to enter a text prompt and generate an
image through a Gradio interface. It includes options for model and
device selection, negative prompts, image dimensions, inference steps,
guidance scale, scheduler selection, random seeds, generation details,
and saving generated images with metadata.

### Base Application Features

-   Generate images from text prompts.
-   Provide negative prompts describing what to avoid.
-   Select among the models supported by the application.
-   Choose automatic device selection, GPU (CUDA), or CPU.
-   Configure image width and height.
-   Configure inference steps and guidance scale.
-   Select a supported scheduler.
-   Use a seed for reproducible generation.
-   View generation time and settings.
-   Save generated images and metadata.
-   View example prompts and recent generated images.
-   Learn about diffusion and its main components through an educational
    section.

The base application is designed primarily for Google Colab with GPU
support. A temporary public Gradio URL can be created when sharing is
enabled; it depends on the running session.

## Additional Work in `Text_to_Image_Merged (1).ipynb`

The notebook develops six related tasks. For the diffusion-based work,
it builds on the existing pretrained Stable Diffusion 1.5 model rather
than treating the project as a new diffusion model trained from scratch.
The other tasks implement separate GAN experiments and text-conditioning
components.

### Task 1 --- Stable Diffusion 1.5 LoRA Fine-Tuning

Applies Low-Rank Adaptation (LoRA) to adapt the existing pretrained
Stable Diffusion 1.5 model toward flower imagery. The notebook includes
flower image/caption preparation, training configuration,
adapter/checkpoint handling, image generation, and comparisons between
the base model and the LoRA-adapted model. It also includes experiments
with different adapter weights.

### Task 2 --- Five-Class Conditional GAN

Explores a separate class-conditional GAN trained on five flower
categories:

-   Daisy
-   Dandelion
-   Rose
-   Sunflower
-   Tulip

The recorded dataset summary contains 2,761 images, and the GAN
configuration uses 64 × 64 images. The generator uses noise and class
information to create images, while the discriminator evaluates real and
generated images with class conditioning.

### Task 3 --- Text Preprocessing and CLIP Embeddings

Uses `openai/clip-vit-base-patch32` to tokenize flower names or
descriptions and obtain text embeddings. Recorded notebook checks show
token-level embeddings and pooled text features. These representations
can provide conditioning information to the text-conditioned GAN; CLIP
is not the image generator itself.

### Task 4 --- Flower Dataset Analysis

Analyzes flower class distributions, image dimensions/resolutions, and
available descriptions or captions. The notebook also prepares and
documents data for the later experiments, including a 40-class subset of
Oxford Flowers 102 with class labels re-indexed to `0–39`.

### Task 5 --- Attention-Based and Text-Conditioned GAN

Explores a separate attention-based GAN on a 40-class Oxford Flowers 102
subset. The architecture includes self-attention and a projection
discriminator. A text-conditioned variant uses CLIP token embeddings and
cross-attention to connect text features with image features.

The notebook contains recorded training and shape-test outputs,
including a 30-epoch attention-GAN run and a five-epoch text-conditioned
experiment. These are GAN experiments, separate from Stable Diffusion.

### Task 6 --- Integrated Prompt-to-Image Workflow

Connects prompt preprocessing, flower-class resolution, text
conditioning, and the trained text-conditioned GAN to generate flower
images. The notebook also includes a separate Stable Diffusion
generation path for higher-resolution outputs. Results from the GAN path
and diffusion path should be evaluated independently.

## How the Existing Stable Diffusion Application Works

1.  The user enters a text prompt and, optionally, a negative prompt.
2.  The prompt is passed to the Stable Diffusion pipeline.
3.  The model uses text conditioning and iterative denoising to generate
    an image.
4.  The application displays the image in the Gradio interface.
5.  Generation details such as image dimensions, inference steps,
    guidance scale, scheduler, seed, device, and elapsed time can be
    displayed.
6.  When saving is enabled, the image and its metadata are stored
    locally.

The LoRA extension adds a flower-domain adaptation to the pretrained
model. The GAN pipeline follows a separate process and should not be
described as part of the diffusion denoising process.

## Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  PyTorch                             Neural networks, GAN training, and
                                      tensor operations

  Hugging Face Diffusers              Stable Diffusion loading and
                                      generation

  Stable Diffusion 1.5                Pretrained text-to-image diffusion
                                      model

  PEFT / LoRA                         Parameter-efficient model
                                      adaptation

  Hugging Face Transformers           CLIP tokenizer and text encoder

  CLIP                                Text embeddings and prompt/class
                                      similarity

  Gradio                              Interactive text-to-image interface

  NumPy and pandas                    Numerical processing and dataset
                                      analysis

  Matplotlib                          Training plots and image
                                      comparisons

  Pillow (PIL)                        Image loading and processing

  Google Colab and Google Drive       Notebook execution, GPU access, and
                                      artifact storage
  -----------------------------------------------------------------------

## Datasets

### Five-Class Flower Dataset

Used for the smaller class-conditional GAN and flower-domain
experiments. The recorded notebook summary reports 2,761 images across
daisy, dandelion, rose, sunflower, and tulip categories. The notebook
describes the flower data as Flickr imagery with Creative Commons
Attribution licensing; verify the original source and license terms
before redistributing the data.

### Oxford Flowers 102 Subset

Used for the attention-based and text-conditioned GAN experiments. The
notebook describes a subset of 40 classes, remapped to labels `0–39`,
with a recorded training setup of 4,855 samples. The subset is not the
same as the full Oxford Flowers 102 dataset. Check the generated
metadata files for the exact selected classes and split.

The base Stable Diffusion application itself does not require a separate
project dataset for ordinary prompt-to-image generation; it uses a
pretrained model. The flower dataset is used for the fine-tuning and GAN
experiments described above.

## Project Files and Output Locations

The primary notebook is:

``` text
Text_to_Image_Merged (1).ipynb
```

The notebook references the following Google Drive project root:

``` text
/content/drive/MyDrive/ElevanceSkills_Project/
```

Important referenced output locations include:

``` text
ElevanceSkills_Project/
├── Task1_Flower_Finetuning/
│   ├── lora_output/
│   └── final_results/
│       └── task2_cgan/
└── outputs/
    ├── task4_dataset_analysis/
    ├── task5_attention_cgan_v2/
    ├── task5_text_cross_attention/
    │   ├── checkpoints/
    │   └── training_history.csv
    └── task6_integrated_pipeline/
        └── generated_samples/
```

Other notebook-referenced files include:

-   `outputs/task5_attention_cgan_v2/class_text_embeddings.pt`
-   `outputs/task5_attention_cgan_40class/dataset_40class_repaired.csv`
-   `outputs/task5_text_cross_attention/checkpoints/final_text_conditioned_gan.pt`
-   Generated samples under
    `outputs/task6_integrated_pipeline/generated_samples/`

These paths reflect locations referenced by the notebook, not a
guarantee that every file currently exists. Some directories are created
only when their corresponding cells run. Confirm the current Drive
contents before relying on a checkpoint or sample.

The notebook also references a flower-image path containing the spelling
`Task1_Flower_Finetuning_datatset`. Preserve the actual folder name in
your environment or update the code and documentation together if it is
changed.

## Running the Project

The notebook and base application are designed primarily for Google
Colab with GPU support.

1.  Open `Text_to_Image_Merged (1).ipynb` in Google Colab or a
    compatible Jupyter environment.
2.  Mount Google Drive if using the recorded `/content/drive/MyDrive/`
    paths.
3.  Verify that the expected datasets, metadata, and output directories
    are available.
4.  Run the setup and import cells, and check the installed versions of
    PyTorch, Diffusers, Transformers, and other dependencies.
5.  Run the dataset preparation and analysis sections before training
    models that depend on them.
6.  Run the LoRA section and verify that the adapter/checkpoint is saved
    before loading it for generation.
7.  Run the CGAN and CLIP sections, checking class IDs and embedding
    dimensions.
8.  Run the attention-GAN and text cross-attention sections and verify
    that checkpoints and logs are written.
9.  Run Task 6 only after confirming that the checkpoint matches the
    exact generator architecture used to train it.
10. Review saved images and logs to verify the results.

For the base Gradio application, start the relevant application code,
initialize a supported model and device, enter a prompt, adjust settings
as needed, and select **Generate Image**.

## Resource and Reproducibility Notes

-   A suitable GPU is recommended for model fine-tuning, GAN training,
    and practical Stable Diffusion generation. CPU execution may be much
    slower.
-   Higher resolution and more inference steps increase compute and
    memory requirements.
-   Use the same prompt and seed when comparing model variants, where
    possible.
-   Keep class-label mappings consistent with each GAN's configured
    number of classes.
-   Load checkpoints only into the architecture and configuration that
    created them.
-   Record library versions, model identifiers, training settings, and
    random seeds when reproducing results.
-   A temporary Gradio share link is not a permanent deployment and
    requires the Colab session to remain active.

## Verification Status and Limitations

The notebook contains implementation code and recorded outputs for
several experiments, but code presence alone does not prove that every
training or generation stage completed successfully.

The project record includes CLIP embedding checks,
attention/cross-attention shape tests, recorded attention-GAN and
text-conditioned GAN training, and sample-generation code. It also
includes issues that should be resolved or rechecked before reporting
the whole workflow as fully validated:

-   A CUDA device-side assertion was encountered in the five-class CGAN
    discriminator's class-projection operation. Check the dataset labels
    and model class count, then restart the runtime before rerunning.
-   An attempted reconstructed Task 6 generator did not match a saved
    checkpoint. Use the exact architecture and configuration from the
    training notebook rather than an approximate reconstruction.
-   Verify that the LoRA adapter, GAN checkpoints, logs, and generated
    samples exist in the current Drive directories.
-   Evaluate the Stable Diffusion and GAN generation routes separately.
    A successful diffusion output does not establish that the GAN route
    is working.
-   Training loss values and tensor-shape tests are useful diagnostics,
    but they do not by themselves prove image quality or prompt
    accuracy.

### Final verification checklist

-   [ ] Base Gradio application initializes and generates an image from
    a prompt.
-   [ ] LoRA adapter exists and loads into the intended Stable Diffusion
    version.
-   [ ] Base-versus-LoRA comparison runs with a consistent prompt and
    seed.
-   [ ] Five-class CGAN loads and generates samples for valid class IDs.
-   [ ] CLIP embeddings have the expected dimensions.
-   [ ] The 40-class dataset mapping matches the trained attention GAN.
-   [ ] Attention-GAN and text-conditioned GAN checkpoints load into
    their original architectures.
-   [ ] Task 6 generates and saves an image through the GAN route.
-   [ ] The separate Stable Diffusion generation path is tested
    independently.
-   [ ] Output files and logs are present and correspond to the runs
    being reported.

## Future Scope

Possible extensions include:

-   Image-to-image generation and inpainting.
-   More precise prompt control and prompt enhancement.
-   Additional pretrained diffusion models.
-   Improved evaluation of generated image quality and prompt alignment.
-   Better experiment tracking and checkpoint metadata.
-   A persistent deployment for the Gradio application.
-   Further GAN training and evaluation after resolving label and
    checkpoint compatibility issues.

## Project Type

**Generative AI / Text-to-Image / Diffusion Model Application / LoRA
Fine-Tuning / Conditional GANs**

## Author

**Zoya Kadodiya**
