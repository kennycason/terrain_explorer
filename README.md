# 3D Terrain Explorer

Explore a procedurally generated world in 3D, built from a height map image.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the explorer:**
   ```bash
   python terrain_explorer.py
   ```

## Controls

| Key | Action |
|-----|--------|
| `W` | Move forward |
| `A` | Move left |
| `S` | Move backward |
| `D` | Move right |
| `Mouse` | Look around |
| `Space` | Jump |
| `Shift` | Run faster |
| `ESC` | Exit |

## How It Works

The terrain explorer reads the `image.png` height map and interprets colors as elevation:

- **Blue** → Water (lowest elevation)
- **Green** → Land (medium elevation)
- **Yellow/Tan** → Hills (higher elevation)
- **Brown/Red** → Mountains (highest elevation)

The image is converted to a 3D mesh that you can walk around and explore.

## Configuration

You can adjust these values in `terrain_explorer.py`:

- `TERRAIN_RESOLUTION` - Higher = more detail, but slower (default: 256)
- `HEIGHT_SCALE` - How tall mountains are (default: 30)
- `TERRAIN_SCALE` - How spread out the terrain is (default: 2)

