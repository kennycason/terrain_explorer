# 🌍 Terrain Explorer

## 📸 Screenshots

| Normalized (Smooth) | Original | Psychedelic 🌈 |
|:---:|:---:|:---:|
| <img src="screenshot_normalized.png" width="100%"> | <img src="screenshot.png" width="100%"> | <img src="screenshot_psychedelic.png" width="100%"> |

---

### 🎬 Demo Videos

| Flight | Destructible Terrain | Normalized |
|:---:|:---:|:---:|
| [▶️ Flight Demo](andrew_maps_03.mp4) | [▶️ Bullets Demo](andrew_maps_04_bullets.mp4) | [▶️ Normalized Demo](andrew_maps_07_normalized.mp4) |

<video src="andrew_maps_03.mp4" width="32%" controls></video>
<video src="andrew_maps_04_bullets.mp4" width="32%" controls></video>
<video src="andrew_maps_07_normalized.mp4" width="32%" controls></video>

---

## 🌈 PSYCHEDELIC MODE 🌈

| 🍄 Psychedelic | 🍄 Psychedelic Normalized |
|:---:|:---:|
| [▶️ Watch Psychedelic](andrew_maps_05_psychedelic.mp4) | [▶️ Watch Psychedelic Normalized](andrew_maps_06_psychedelic_normalized.mp4) |

<video src="andrew_maps_05_psychedelic.mp4" width="49%" controls></video>
<video src="andrew_maps_06_psychedelic_normalized.mp4" width="49%" controls></video>

Run the psychedelic version for a **trippy experience**:
```bash
python terrain_explorer_psychedelic.py
```

**Features:**
- 🎨 Rainbow terrain colors that shift over time
- 🌅 Cycling sky & fog colors
- 💫 Vertical breathing terrain (pulses up/down)
- 🦠 **Fractal cancer growth** - terrain permanently morphs with spreading tendrils!
- 🌈 Rainbow projectiles with trails
- 💥 Psychedelic explosions with sparkles
- ✨ Everything cycles through the color spectrum

---

## 🧠 GAN GENERATOR MODE

Integrate with Andrew's terrain generator to create **infinite unique maps**!

```bash
# Requires PyTorch + the generator model
pip install torch

# Place generator_v1.pt in runs/models/ or current directory
python terrain_explorer_generator.py
```

**Controls:**
- **G** = Generate a NEW random map!
- All other controls same as normal mode

The generator creates procedural heightmaps using a trained GAN model. Each press of G creates a completely new world to explore!

---

A 3D terrain explorer with **dynamic chunk loading**, **jet flight physics**, and **destructible terrain**. Built with Python, Pygame, and OpenGL.

## ✨ Features

- 🗺️ **Dynamic Chunk Loading** - Terrain loads/unloads as you explore
- ✈️ **Jet Flight Mode** - Full 360° aerobatics, do loops and barrel rolls!
- 💥 **Destructible Terrain** - Fire projectiles to blast craters in the world
- 🧭 **Live Minimap** - See the whole world and your position in real-time
- 🌊 **Realistic Coloring** - Ocean depths, beaches, forests, mountains, snow peaks

## 🚀 Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the explorer
python terrain_explorer.py
```

## 🎮 Controls

### Movement (Left Stick)
| Key | Action |
|-----|--------|
| `W` / `↑` | Fly forward (in facing direction) |
| `S` / `↓` | Fly backward |
| `A` / `←` | Strafe left |
| `D` / `→` | Strafe right |
| `H` | Fly UP |
| `F` | Fly DOWN |
| `Alt` | Move faster |

### Camera (Right Stick)
| Key | Action |
|-----|--------|
| `I` | Pitch UP (hold to do loops!) |
| `K` | Pitch DOWN |
| `J` | Look LEFT |
| `L` | Look RIGHT |
| `Right-Click + Mouse` | Free look |

### Weapons
| Key | Action |
|-----|--------|
| `Space` | 🔥 Fire projectile |
| `Left-Click` | 🔥 Fire projectile (alt) |

*3 shots per second • Creates craters • Updates minimap in real-time*

### Other
| Key | Action |
|-----|--------|
| `ESC` | Exit |

## 💣 Destructible Terrain

Fire projectiles at the terrain to:
- **Blast craters** with realistic spherical shapes
- **Lower terrain** - reduce mountain heights
- **Create lakes** - blast below sea level to fill with water
- **See destruction on minimap** - updates live!

## 🗺️ How It Works

The explorer reads a raw NumPy heightmap file (`.npy`) with elevation data:

- **Negative values** → Ocean (deeper = darker blue)
- **~0** → Sea level / beaches
- **Positive values** → Land (green → brown → snow peaks)

Terrain is split into 64×64 chunks that load dynamically as you fly around, allowing exploration of massive worlds.

## ⚙️ Configuration

Adjust these values in `terrain_explorer.py`:

```python
CHUNK_SIZE = 64           # Size of each terrain chunk
CHUNK_RENDER_DISTANCE = 3 # How many chunks to render
HEIGHT_SCALE = 3.5        # Vertical exaggeration (smoothed)
TERRAIN_SCALE = 0.8       # Horizontal scale
EXPLOSION_RADIUS = 15     # Crater size
FIRE_RATE = 0.33          # Shots per second (3)
```

## 📁 Files

- `terrain_explorer.py` - Main application
- `terrain_explorer_psychedelic.py` - 🌈 Trippy version!
- `terrain_explorer_generator.py` - 🧠 GAN generator integration!
- `demo.py` - Andrew's Gradio generator UI
- `raw_map_*.npy` - Raw heightmap data
- `requirements.txt` - Python dependencies
- `screenshot.png` - Original preview
- `screenshot_normalized.png` - Smoothed terrain preview
- `screenshot_psychedelic.png` - Psychedelic preview

## 🎯 Tips

- **Do a loop**: Hold `I` while flying forward with `W`
- **Strafe run**: Combine `A`/`D` with `J`/`L` for cinematic flight
- **Terraform**: Blast mountains flat or create new lakes!
- **Check minimap**: Red square shows your position on the full world map

---

Made with 🐍 Python + 🎮 Pygame + 🔺 OpenGL
