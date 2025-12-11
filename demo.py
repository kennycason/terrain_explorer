import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import gradio as gr

from visualization import create_image_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache for loaded TorchScript generator modules, keyed by path
_cached_models: Dict[str, torch.jit.ScriptModule] = {}


def _find_generator_checkpoints(models_dir: str = "runs/models") -> List[str]:
    """
    Scan the models directory for exported generator checkpoints.
    """
    if not os.path.isdir(models_dir):
        return []

    files = []
    for name in os.listdir(models_dir):
        if name.endswith(".pt"):
            files.append(os.path.join(models_dir, name))

    # Sort newest first for convenience
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _format_model_choices(choices: List[str]) -> List[tuple[str, str]]:
    """
    Format model paths for display in dropdown, showing only the filename.
    Returns list of tuples: (display_label, full_path)
    """
    return [(os.path.basename(path), path) for path in choices]


def _load_model(model_path: str) -> torch.jit.ScriptModule:
    """
    Lazily load a TorchScript generator from disk, caching by path.
    """
    if model_path in _cached_models:
        return _cached_models[model_path]

    module = torch.jit.load(model_path, map_location=device)
    module.eval()
    _cached_models[model_path] = module
    return module


def _delete_model(model_path: str) -> tuple[gr.Dropdown, str]:
    """
    Delete a model file and remove it from cache.
    Returns updated dropdown and status message.
    """
    if not model_path:
        return gr.Dropdown(), "No model selected to delete."
    
    if not os.path.exists(model_path):
        return gr.Dropdown(), f"Model file not found: {model_path}"
    
    try:
        # Remove from cache if loaded
        if model_path in _cached_models:
            del _cached_models[model_path]
        
        # Delete the file
        os.remove(model_path)
        
        # Refresh model list
        choices = _find_generator_checkpoints()
        formatted_choices = _format_model_choices(choices)
        value = choices[0] if choices else None
        
        return gr.Dropdown(choices=formatted_choices, value=value), f"Deleted: {os.path.basename(model_path)}"
    except Exception as e:
        return gr.Dropdown(), f"Error deleting model: {str(e)}"


def _mark_model_as_good(model_path: str) -> tuple[gr.Dropdown, str]:
    """
    Rename a model to mark it as approved (generate-approved-<date>).
    Returns updated dropdown and status message.
    """
    if not model_path:
        return gr.Dropdown(), "No model selected to mark as good."
    
    if not os.path.exists(model_path):
        return gr.Dropdown(), f"Model file not found: {model_path}"
    
    try:
        # Get directory and current filename
        model_dir = os.path.dirname(model_path)
        current_name = os.path.basename(model_path)
        
        # Generate new name with date
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"generate-approved-{date_str}.pt"
        new_path = os.path.join(model_dir, new_name)
        
        # Check if new name already exists (unlikely but possible)
        if os.path.exists(new_path):
            # Add a counter
            counter = 1
            while os.path.exists(new_path):
                new_name = f"generate-approved-{date_str}_{counter}.pt"
                new_path = os.path.join(model_dir, new_name)
                counter += 1
        
        # Remove from cache if loaded
        if model_path in _cached_models:
            del _cached_models[model_path]
        
        # Rename the file
        os.rename(model_path, new_path)
        
        # Refresh model list
        choices = _find_generator_checkpoints()
        formatted_choices = _format_model_choices(choices)
        value = new_path if new_path in choices else (choices[0] if choices else None)
        
        return gr.Dropdown(choices=formatted_choices, value=value), f"Marked as good: {new_name}"
    except Exception as e:
        return gr.Dropdown(), f"Error marking model as good: {str(e)}"


@torch.inference_mode()
def generate_samples(
    model_path: str,
    c_priority: float,
    avg_z_priority: float,
    std_z_priority: float,
    d_priority: float,
    dn_priority_1: float,
    dn_priority_2: float,
    dn_priority_3: float,
    dn_priority_4: float,
    dn_priority_5: float,
    c_opinion_priority: float,
    avg_z: float,
    std_z: float,
    batch_size: int,
    seed: int | None,
    scale: float = 1.0,
):
    """
    Generate a grid of samples from an exported Z0 generator, visualized with the terrain colormap.
    
    The generator expects 8 separate tensor arguments (all values in [0, 1] range):
    - c_priorities: C priority (batch_size, 1)
    - avg_z_priorities: Average z priority (batch_size, 1)
    - std_z_priorities: Standard deviation z priority (batch_size, 1)
    - d_priorities: Diversity priority (batch_size, 1)
    - dn_priorities: Diversity-normalized priorities (batch_size, 5)
    - c_opinion_priorities: Critical opinion priority (batch_size, 1)
    - avg_z_parameters: Average z parameter (batch_size, 1)
    - std_z_parameters: Standard deviation z parameter (batch_size, 1)
    
    The model internally handles priority normalization and denormalization to the output range.
    """

    if not model_path:
        raise gr.Error("No generator checkpoint selected. Please choose a model file.")

    model = _load_model(model_path)

    # Optional deterministic seeding for reproducible sampling
    if seed is not None:
        seed = int(seed)
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # Build individual tensor inputs matching the generator's forward() signature
    # All values should be in [0, 1] range - the model handles denormalization internally
    c_priorities = torch.tensor([[c_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    avg_z_priorities = torch.tensor([[avg_z_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    std_z_priorities = torch.tensor([[std_z_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    d_priorities = torch.tensor([[d_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    dn_priorities = torch.tensor(
        [[dn_priority_1, dn_priority_2, dn_priority_3, dn_priority_4, dn_priority_5]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    c_opinion_priorities = torch.tensor([[c_opinion_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    avg_z_parameters = torch.tensor([[avg_z]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    std_z_parameters = torch.tensor([[std_z]], dtype=torch.float32, device=device).repeat(batch_size, 1)

    # Generator samples all of its own randomness internally.
    # The exported model expects individual tensor arguments matching the forward() signature
    outputs = model(
        c_priorities,
        avg_z_priorities,
        std_z_priorities,
        d_priorities,
        dn_priorities,
        c_opinion_priorities,
        avg_z_parameters,
        std_z_parameters,
    )
    
    # Access the output which is the terrain (B, 1, H, W)
    images = outputs

    # Use the full batch as the grid size so batch size and grid images match.
    n_images = images.shape[0]
    # Pass the scale as upscale_factor to create_image_grid
    grid = create_image_grid(images, n_images=n_images, upscale_factor=int(scale))

    # Convert from (3, H, W) with float values in [0, 1] to (H, W, 3) uint8 for Gradio
    grid = grid.clamp(0.0, 1.0)
    grid = (grid * 255.0).byte().permute(1, 2, 0).cpu().numpy()

    return grid


@torch.inference_mode()
def save_raw_map(
    model_path: str,
    c_priority: float,
    avg_z_priority: float,
    std_z_priority: float,
    d_priority: float,
    dn_priority_1: float,
    dn_priority_2: float,
    dn_priority_3: float,
    dn_priority_4: float,
    dn_priority_5: float,
    c_opinion_priority: float,
    avg_z: float,
    std_z: float,
    batch_size: int,
    seed: int | None,
    output_path: str | None = None,
) -> str:
    """
    Generate samples and save the raw floating point map data as a numpy file.
    
    Returns the path to the saved file.
    """
    if not model_path:
        raise gr.Error("No generator checkpoint selected. Please choose a model file.")

    model = _load_model(model_path)

    # Optional deterministic seeding for reproducible sampling
    if seed is not None:
        seed = int(seed)
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # Build individual tensor inputs matching the generator's forward() signature
    c_priorities = torch.tensor([[c_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    avg_z_priorities = torch.tensor([[avg_z_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    std_z_priorities = torch.tensor([[std_z_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    d_priorities = torch.tensor([[d_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    dn_priorities = torch.tensor(
        [[dn_priority_1, dn_priority_2, dn_priority_3, dn_priority_4, dn_priority_5]],
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    c_opinion_priorities = torch.tensor([[c_opinion_priority]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    avg_z_parameters = torch.tensor([[avg_z]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    std_z_parameters = torch.tensor([[std_z]], dtype=torch.float32, device=device).repeat(batch_size, 1)

    # Generate the raw map data
    outputs = model(
        c_priorities,
        avg_z_priorities,
        std_z_priorities,
        d_priorities,
        dn_priorities,
        c_opinion_priorities,
        avg_z_parameters,
        std_z_parameters,
    )
    
    # Access the output which is the terrain (B, 1, H, W)
    images = outputs

    # Convert to numpy array (B, 1, H, W) -> (B, H, W) and move to CPU
    raw_data = images.squeeze(1).cpu().numpy().astype(np.float32)

    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"raw_map_{timestamp}.npy"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save as numpy file
    np.save(output_path, raw_data)

    # Return absolute path for Gradio file serving
    return os.path.abspath(output_path)


def build_interface():
    """
    Construct a simple Gradio UI for exploring the generator.
    """
    # Custom CSS to disable image transitions and make UI more compact
    custom_css = """
    .gradio-image img {
        transition: none !important;
        image-rendering: pixelated !important;
        image-rendering: -moz-crisp-edges !important;
        image-rendering: crisp-edges !important;
    }
    .gradio-image {
        transition: none !important;
    }
    /* Make sliders and form elements more compact */
    .form-item {
        margin-bottom: 0.5rem !important;
    }
    .gradio-group {
        padding: 0.75rem !important;
    }
    /* Reduce vertical spacing */
    .block {
        margin-bottom: 0.5rem !important;
    }
    /* Make controls column narrower */
    .controls-column {
        max-width: 400px !important;
    }
    """
    
    with gr.Blocks(title="Z0 Generator Explorer", css=custom_css) as demo:
        gr.Markdown(
            """
            **Z0 Generator Explorer**

            Use the sliders to adjust the **priorities**:
            - **C Priority**: The first priority
            - **Avg Z Priority**: Average z priority
            - **Std Z Priority**: Standard deviation z priority
            - **D Priority**: Diversity priority (scales the DN priorities)
            - **DN Priority 1-5**: Five diversity-normalized priorities (scaled by D Priority)
            - **C Opinion Priority**: Critical opinion priority (in [0, 1] range)
            - **Avg Z**: Average z parameter (in [0, 1] range)
            - **Std Z**: Standard deviation z parameter (in [0, 1] range)
            
            Choose the **batch size / grid size** and an optional **random seed**. The image updates automatically as you adjust the sliders.
            
            All priority and parameter values should be in the [0, 1] range. The model handles normalization and denormalization internally.
            """
        )

        # Main layout: controls on left, image on right
        with gr.Row():
            # Left column: Controls (narrow)
            with gr.Column(scale=1, min_width=350):
                # Seed navigation buttons (always visible)
                with gr.Row():
                    prev_seed_button = gr.Button("Previous", variant="secondary", scale=1)
                    next_seed_button = gr.Button("Next", variant="secondary", scale=1)
                
                # Use Tabs to organize controls
                with gr.Tabs():
                    # Priorities tab
                    with gr.Tab("Priorities"):
                        c_priority = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.01,
                            label="C Priority",
                        )
                        avg_z_priority = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="Avg Z Priority",
                        )
                        std_z_priority = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="Std Z Priority",
                        )
                        d_priority = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="D Priority",
                        )
                        dn_priority_1 = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="DN Priority 1",
                        )
                        dn_priority_2 = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="DN Priority 2",
                        )
                        dn_priority_3 = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="DN Priority 3",
                        )
                        dn_priority_4 = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="DN Priority 4",
                        )
                        dn_priority_5 = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="DN Priority 5",
                        )
                        c_opinion_priority = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="C Opinion Priority",
                        )

                    # Z Constraints tab (non-priority settings)
                    with gr.Tab("Z Constraints"):
                        avg_z = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="Avg Z",
                        )
                        std_z = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="Std Z",
                        )

                    # Generation Settings tab
                    with gr.Tab("Generation Settings"):
                        # Model selection (compact)
                        model_choices = _find_generator_checkpoints()
                        default_model = model_choices[0] if model_choices else None
                        formatted_choices = _format_model_choices(model_choices)
                        default_value = default_model if default_model else None

                        model_dropdown = gr.Dropdown(
                            choices=formatted_choices,
                            value=default_value,
                            label="Model",
                            interactive=True,
                        )
                        
                        # Compact button row
                        with gr.Row():
                            refresh_models = gr.Button("Refresh", variant="secondary", scale=1)
                            mark_as_good_button = gr.Button("Mark Good", variant="secondary", scale=1)
                            delete_model_button = gr.Button("Delete", variant="stop", scale=1)
                            generate_button = gr.Button("Generate", variant="primary", scale=1)
                        
                        # Save raw map controls
                        save_raw_button = gr.Button("Save Raw Map", variant="secondary")
                        output_path_input = gr.Textbox(
                            label="Output Path (optional, leave empty for auto-generated name)",
                            placeholder="e.g., maps/my_map.npy",
                            visible=True,
                        )
                        save_file_download = gr.File(
                            label="Download Raw Map",
                            visible=True,
                            interactive=False,
                        )
                        save_status = gr.Textbox(label="Save Status", interactive=False, visible=True)
                        
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=1,
                            step=1,
                            label="Batch Size",
                        )
                        seed = gr.Slider(
                            minimum=0,
                            maximum=2**31 - 1,
                            value=1,
                            step=1,
                            label="Random Seed (0 = random)",
                        )
                        image_scale = gr.Slider(
                            minimum=1.0,
                            maximum=8.0,
                            value=8.0,
                            step=0.5,
                            label="Image Scale",
                        )

            # Right column: Image (wide)
            with gr.Column(scale=3):
                output_image = gr.Image(
                    label="Generated Terrain Grid",
                    type="numpy",
                    height=800,
                )

        # Refresh button updates the dropdown choices and selects the latest model, then triggers generation
        def _refresh_model_choices_and_generate(
            current_model: str | None,
            c_pri: float,
            avg_z_pri: float,
            std_z_pri: float,
            d_pri: float,
            dn_pri_1: float,
            dn_pri_2: float,
            dn_pri_3: float,
            dn_pri_4: float,
            dn_pri_5: float,
            c_opinion: float,
            avg_z_val: float,
            std_z_val: float,
            batch: int,
            seed_val: int | None,
            img_scale: float,
        ) -> tuple[gr.Dropdown, Any]:
            choices = _find_generator_checkpoints()
            formatted_choices = _format_model_choices(choices)
            # Always select the latest model (first in list, sorted newest first)
            latest_model = choices[0] if choices else None
            updated_dropdown = gr.Dropdown(choices=formatted_choices, value=latest_model)
            
            # Generate image with the latest model
            if not latest_model:
                raise gr.Error("No generator checkpoints found. Please train a model first.")
            
            image = generate_samples(
                latest_model, c_pri, avg_z_pri, std_z_pri, d_pri, dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5,
                c_opinion, avg_z_val, std_z_val, batch, seed_val, img_scale
            )
            
            return updated_dropdown, image

        refresh_models.click(
            fn=_refresh_model_choices_and_generate,
            inputs=[model_dropdown, c_priority, avg_z_priority, std_z_priority, d_priority, dn_priority_1, dn_priority_2, dn_priority_3, dn_priority_4, dn_priority_5, c_opinion_priority, avg_z, std_z, batch_size, seed, image_scale],
            outputs=[model_dropdown, output_image],
            show_progress=False,
        )
        
        # Delete model button
        def delete_model_and_refresh(model_path: str) -> gr.Dropdown:
            dropdown, message = _delete_model(model_path)
            return dropdown
        
        delete_model_button.click(
            fn=delete_model_and_refresh,
            inputs=[model_dropdown],
            outputs=[model_dropdown],
        )
        
        # Mark as good button
        def mark_good_and_refresh(model_path: str) -> gr.Dropdown:
            dropdown, message = _mark_model_as_good(model_path)
            return dropdown
        
        mark_as_good_button.click(
            fn=mark_good_and_refresh,
            inputs=[model_dropdown],
            outputs=[model_dropdown],
        )

        # Function to handle generation (used by both button and sliders)
        def trigger_generation(*args):
            return generate_samples(*args)

        # Common inputs list for all event handlers
        common_inputs = [model_dropdown, c_priority, avg_z_priority, std_z_priority, d_priority, dn_priority_1, dn_priority_2, dn_priority_3, dn_priority_4, dn_priority_5, c_opinion_priority, avg_z, std_z, batch_size, seed, image_scale]

        # Save raw map button handler
        def save_raw_map_handler(
            model_path: str,
            c_pri: float,
            avg_z_pri: float,
            std_z_pri: float,
            d_pri: float,
            dn_pri_1: float,
            dn_pri_2: float,
            dn_pri_3: float,
            dn_pri_4: float,
            dn_pri_5: float,
            c_opinion: float,
            avg_z_val: float,
            std_z_val: float,
            batch: int,
            seed_val: int | None,
            output_path_arg: str | None,
        ):
            try:
                output_path = save_raw_map(
                    model_path, c_pri, avg_z_pri, std_z_pri, d_pri,
                    dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5,
                    c_opinion, avg_z_val, std_z_val, batch, seed_val,
                    output_path=output_path_arg if output_path_arg else None
                )
                # Return file path for download and status message
                return output_path, f"Saved raw map to: {output_path}"
            except Exception as e:
                # Return None for file (no download) and error message
                return None, f"Error saving raw map: {str(e)}"
        
        save_raw_button.click(
            fn=save_raw_map_handler,
            inputs=common_inputs[:-1] + [output_path_input],  # All inputs except image_scale, plus output_path_input
            outputs=[save_file_download, save_status],
            show_progress=True,
        )

        # Update image when button is clicked
        generate_button.click(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )

        # Update image automatically when sliders change (no animation, instant swap)
        c_priority.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        avg_z_priority.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        std_z_priority.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        d_priority.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        dn_priority_1.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        dn_priority_2.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        dn_priority_3.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        dn_priority_4.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        dn_priority_5.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        c_opinion_priority.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        avg_z.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        std_z.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        batch_size.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        seed.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        image_scale.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )
        model_dropdown.change(
            fn=trigger_generation,
            inputs=common_inputs,
            outputs=[output_image],
            show_progress=False,
        )

        # Next/Previous seed navigation
        def next_seed(current_seed, model, c_pri, avg_z_pri, std_z_pri, d_pri, dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5, c_opinion, avg_z_val, std_z_val, batch, img_scale):
            """Increment seed by 1 and generate."""
            new_seed = int(current_seed) + 1
            # Clamp to valid range
            new_seed = max(1, min(new_seed, 2**31 - 1))
            # Generate with new seed
            result = trigger_generation(model, c_pri, avg_z_pri, std_z_pri, d_pri, dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5, c_opinion, avg_z_val, std_z_val, batch, new_seed, img_scale)
            return new_seed, result
        
        def prev_seed(current_seed, model, c_pri, avg_z_pri, std_z_pri, d_pri, dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5, c_opinion, avg_z_val, std_z_val, batch, img_scale):
            """Decrement seed by 1 and generate."""
            new_seed = int(current_seed) - 1
            # Clamp to valid range (minimum 1, since 0 means random)
            new_seed = max(1, new_seed)
            # Generate with new seed
            result = trigger_generation(model, c_pri, avg_z_pri, std_z_pri, d_pri, dn_pri_1, dn_pri_2, dn_pri_3, dn_pri_4, dn_pri_5, c_opinion, avg_z_val, std_z_val, batch, new_seed, img_scale)
            return new_seed, result
        
        # Navigation inputs (all except seed)
        nav_inputs = [model_dropdown, c_priority, avg_z_priority, std_z_priority, d_priority, dn_priority_1, dn_priority_2, dn_priority_3, dn_priority_4, dn_priority_5, c_opinion_priority, avg_z, std_z, batch_size, image_scale]
        
        next_seed_button.click(
            fn=next_seed,
            inputs=[seed] + nav_inputs,
            outputs=[seed, output_image],
            show_progress=False,
        )
        
        prev_seed_button.click(
            fn=prev_seed,
            inputs=[seed] + nav_inputs,
            outputs=[seed, output_image],
            show_progress=False,
        )

    return demo


def main() -> None:
    """
    Entry point: launch the Gradio app.
    """
    demo = build_interface()
    # Share can be toggled if you want external access; by default serve locally.
    demo.launch()


if __name__ == "__main__":
    main()


