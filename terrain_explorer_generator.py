"""
3D Terrain Explorer with Andrew's Generator Integration
Generate new maps with the GAN model and explore them in 3D!

Controls:
  - Arrow Keys or WASD: Move in facing direction
  - Hold Right Mouse: Look around
  - H/F: Fly up/down
  - Space: Fire projectile!
  - G: Generate NEW random map!
  - Escape: Exit
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import GLubyte
from OpenGL.GLU import *
import numpy as np
import math
import os

# Check if torch is available for generator
TORCH_AVAILABLE = False
torch = None
try:
    import torch as _torch
    torch = _torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. Generator features disabled.")
    print("Install with: pip install torch")

# Configuration
CHUNK_SIZE = 64
CHUNK_RENDER_DISTANCE = 3
HEIGHT_SCALE = 3.5
TERRAIN_SCALE = 0.8
MOVE_SPEED = 1.2
MOUSE_SENSITIVITY = 0.06

# Projectile settings
BULLET_SPEED = 3.0
BULLET_LIFETIME = 10.0
EXPLOSION_RADIUS = 15
FIRE_RATE = 0.33

# Generator settings - prefer CPU-patched version
DEFAULT_MODEL_PATH = "generator_v1_cpu.pt"
GENERATOR_DEVICE = None  # Will be set on init


# Only define TerrainGenerator if torch is available
if TORCH_AVAILABLE:
    class TerrainGenerator:
        """Wrapper for Andrew's TorchScript terrain generator"""
        
        def __init__(self, model_path=DEFAULT_MODEL_PATH):
            global GENERATOR_DEVICE
            
            GENERATOR_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Generator device: {GENERATOR_DEVICE}")
            
            if not os.path.exists(model_path):
                # Try alternate locations - prefer CPU version if available
                alt_paths = [
                    "generator_v1_cpu.pt",  # CPU-patched version first
                    "generator_v1.pt",
                    "runs/models/generator_v1_cpu.pt",
                    "runs/models/generator_v1.pt",
                    "../runs/models/generator_v1.pt",
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        model_path = alt
                        break
                else:
                    raise FileNotFoundError(f"Generator model not found: {model_path}")
            
            print(f"  Loading generator from: {model_path}")
            self.model = torch.jit.load(model_path, map_location=GENERATOR_DEVICE)
            self.model.eval()
            print("  Generator loaded successfully!")
        
        def generate(
            self,
            c_priority=1.0,
            avg_z_priority=0.5,
            std_z_priority=0.5,
            d_priority=0.5,
            dn_priorities=(0.5, 0.5, 0.5, 0.5, 0.5),
            c_opinion_priority=0.5,
            avg_z=0.5,
            std_z=0.5,
            seed=None,
        ):
            """Generate a terrain heightmap using the GAN"""
            
            with torch.inference_mode():
                if seed is not None and seed != 0:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                
                # Build input tensors (batch size 1)
                device = GENERATOR_DEVICE
                
                c_priorities = torch.tensor([[c_priority]], dtype=torch.float32, device=device)
                avg_z_priorities = torch.tensor([[avg_z_priority]], dtype=torch.float32, device=device)
                std_z_priorities = torch.tensor([[std_z_priority]], dtype=torch.float32, device=device)
                d_priorities = torch.tensor([[d_priority]], dtype=torch.float32, device=device)
                dn_priorities_tensor = torch.tensor([list(dn_priorities)], dtype=torch.float32, device=device)
                c_opinion_priorities = torch.tensor([[c_opinion_priority]], dtype=torch.float32, device=device)
                avg_z_parameters = torch.tensor([[avg_z]], dtype=torch.float32, device=device)
                std_z_parameters = torch.tensor([[std_z]], dtype=torch.float32, device=device)
                
                # Generate!
                output = self.model(
                    c_priorities,
                    avg_z_priorities,
                    std_z_priorities,
                    d_priorities,
                    dn_priorities_tensor,
                    c_opinion_priorities,
                    avg_z_parameters,
                    std_z_parameters,
                )
                
                # Convert to numpy (B, 1, H, W) -> (H, W)
                heightmap = output.squeeze().cpu().numpy().astype(np.float32)
            
            return heightmap
else:
    TerrainGenerator = None


def smooth_heightmap(data, iterations=2, kernel_size=3):
    """Apply smoothing to reduce spikiness"""
    try:
        from scipy.ndimage import uniform_filter
        smoothed = data.copy()
        for _ in range(iterations):
            smoothed = uniform_filter(smoothed, size=kernel_size, mode='nearest')
        return smoothed
    except ImportError:
        print("Warning: scipy not found, skipping smoothing")
        return data


def load_heightmap(filename, smooth=True):
    """Load a .npy heightmap file"""
    print("Loading heightmap...")
    data = np.load(filename)
    
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    
    print(f"  Shape: {data.shape}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")
    
    if smooth:
        print("  Applying smoothing...")
        data = smooth_heightmap(data, iterations=2, kernel_size=3)
    
    return data.astype(np.float32)


# Player height above terrain when walking
PLAYER_HEIGHT = 3.0
WALK_SMOOTH_SPEED = 0.15  # How smoothly camera follows terrain


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0
        self.pitch = -20
        self.flying = True  # Start in fly mode
        self.target_y = 40  # For smooth walking
        
    def rotate(self, dx, dy):
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY
        while self.pitch > 180: self.pitch -= 360
        while self.pitch < -180: self.pitch += 360
        
    def rotate_keyboard(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch += dpitch
        while self.pitch > 180: self.pitch -= 360
        while self.pitch < -180: self.pitch += 360
    
    def get_terrain_height(self, heightmap, terrain_scale, height_scale):
        """Get terrain height at current position"""
        h, w = heightmap.shape
        map_x = int(self.x / terrain_scale + w / 2)
        map_z = int(self.z / terrain_scale + h / 2)
        
        if 0 <= map_x < w and 0 <= map_z < h:
            # Bilinear interpolation for smoother walking
            fx = self.x / terrain_scale + w / 2
            fz = self.z / terrain_scale + h / 2
            
            x0, z0 = int(fx), int(fz)
            x1, z1 = min(x0 + 1, w - 1), min(z0 + 1, h - 1)
            
            # Fractional parts
            tx = fx - x0
            tz = fz - z0
            
            # Sample 4 corners
            h00 = heightmap[z0, x0]
            h10 = heightmap[z0, x1]
            h01 = heightmap[z1, x0]
            h11 = heightmap[z1, x1]
            
            # Bilinear interpolation
            h_interp = (h00 * (1-tx) * (1-tz) + 
                       h10 * tx * (1-tz) + 
                       h01 * (1-tx) * tz + 
                       h11 * tx * tz)
            
            return h_interp * height_scale
        return 0
        
    def move(self, forward, right, up, heightmap, terrain_scale, height_scale):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        if self.flying:
            # Full 3D movement in fly mode
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        else:
            # Horizontal movement only in walk mode
            forward_x = -math.sin(yaw_rad)
            forward_y = 0
            forward_z = -math.cos(yaw_rad)
        
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Apply horizontal movement
        self.x += (forward * forward_x + right * right_x) * MOVE_SPEED
        self.z += (forward * forward_z + right * right_z) * MOVE_SPEED
        
        # Get terrain height at new position
        terrain_h = self.get_terrain_height(heightmap, terrain_scale, height_scale)
        min_height = terrain_h + PLAYER_HEIGHT
        
        if self.flying:
            # In fly mode, apply vertical movement
            up_y = 1.0  # Vertical up/down
            self.y += up * up_y * MOVE_SPEED
            
            # H raises us up (already flying)
            # F lowers us - if we hit ground, switch to walk mode
            if up < 0:  # Pressing F (down)
                if self.y <= min_height:
                    self.y = min_height
                    self.flying = False
                    self.target_y = min_height
            
            # Still enforce minimum height even in fly mode
            if self.y < min_height:
                self.y = min_height
        else:
            # Walk mode - smoothly follow terrain
            self.target_y = min_height
            self.y += (self.target_y - self.y) * WALK_SMOOTH_SPEED
            
            # H key lifts off into fly mode
            if up > 0:
                self.flying = True
                self.y += up * MOVE_SPEED
        
    def apply(self):
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    def get_chunk_pos(self):
        chunk_world_size = CHUNK_SIZE * TERRAIN_SCALE
        return (int(self.x / chunk_world_size), int(self.z / chunk_world_size))
    
    def get_forward_vector(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        return (
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        )


class Projectile:
    def __init__(self, x, y, z, dx, dy, dz):
        self.x, self.y, self.z = x, y, z
        self.dx = dx * BULLET_SPEED
        self.dy = dy * BULLET_SPEED
        self.dz = dz * BULLET_SPEED
        self.lifetime = BULLET_LIFETIME
        self.active = True
    
    def update(self, dt, heightmap, terrain_scale, height_scale):
        if not self.active:
            return None
        
        self.x += self.dx * dt
        self.y += self.dy * dt
        self.z += self.dz * dt
        self.lifetime -= dt / 60.0
        
        if self.lifetime <= 0:
            self.active = False
            return None
        
        h, w = heightmap.shape
        map_x = int(self.x / terrain_scale + w / 2)
        map_z = int(self.z / terrain_scale + h / 2)
        
        if 0 <= map_x < w and 0 <= map_z < h:
            terrain_height = heightmap[map_z, map_x] * height_scale
            if self.y <= terrain_height:
                self.active = False
                return (self.x, self.y, self.z, map_x, map_z)
        
        if self.y < -50 or abs(self.x) > w * terrain_scale or abs(self.z) > h * terrain_scale:
            self.active = False
        
        return None
    
    def draw(self):
        if not self.active:
            return
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.8, 0.2)
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.5, 8, 8)
        gluDeleteQuadric(quadric)
        glEnable(GL_LIGHTING)
        glPopMatrix()


class Explosion:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.lifetime = 2.5
        self.max_lifetime = 2.5
        self.active = True
    
    def update(self, dt):
        if self.active:
            self.lifetime -= dt / 60.0
            if self.lifetime <= 0:
                self.active = False
    
    def draw(self):
        if not self.active:
            return
        
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        size = EXPLOSION_RADIUS * TERRAIN_SCALE * (0.8 + progress * 2.0)
        alpha = 1.0 - progress * 0.8
        
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        for i in range(5):
            ring_progress = (progress + i * 0.12) % 1.0
            ring_size = size * (0.2 + ring_progress * 0.8)
            ring_alpha = alpha * (1.0 - ring_progress * 0.9)
            
            if ring_progress < 0.3:
                r, g, b = 1.0, 1.0, 0.8
            elif ring_progress < 0.6:
                r, g, b = 1.0, 0.6, 0.2
            else:
                r, g, b = 1.0, 0.3, 0.1
            
            glColor4f(r, g, b, ring_alpha * 0.6)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, ring_size * 0.2, 0)
            for j in range(25):
                angle = 2 * math.pi * j / 24
                glVertex3f(math.cos(angle) * ring_size, 
                          math.sin(angle * 2) * ring_size * 0.4,
                          math.sin(angle) * ring_size)
            glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()


class ChunkManager:
    def __init__(self, heightmap):
        self.heightmap = heightmap
        self.height, self.width = heightmap.shape
        self.chunks = {}
        self.h_min = heightmap.min()
        self.h_max = heightmap.max()
        self.chunks_x = (self.width + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.chunks_z = (self.height + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        print(f"  World size: {self.width}x{self.height}")
        print(f"  Chunk grid: {self.chunks_x}x{self.chunks_z} chunks")
    
    def height_to_color(self, h):
        if h < -4: return (0.05, 0.18, 0.5)
        elif h < -1:
            t = (h + 4) / 3
            return (0.08 + t*0.12, 0.28 + t*0.17, 0.55 + t*0.1)
        elif h < 0: return (0.18, 0.48, 0.68)
        elif h < 0.3: return (0.78, 0.72, 0.52)
        elif h < 1.0:
            t = (h - 0.3) / 0.7
            return (0.28 + t*0.02, 0.58 - t*0.08, 0.22)
        elif h < 2.0: return (0.22, 0.48, 0.18)
        elif h < 3.5:
            t = (h - 2.0) / 1.5
            return (0.38 + t*0.18, 0.38 - t*0.08, 0.2)
        elif h < 5.0:
            t = (h - 3.5) / 1.5
            return (0.52 + t*0.13, 0.45 + t*0.1, 0.35 + t*0.15)
        else:
            t = min(1, (h - 5.0) / 1.5)
            return (0.75 + t*0.2, 0.75 + t*0.2, 0.78 + t*0.17)
    
    def create_chunk(self, cx, cz):
        start_x = cx * CHUNK_SIZE + self.width // 2
        start_z = cz * CHUNK_SIZE + self.height // 2
        
        if start_x < 0 or start_z < 0: return None
        if start_x >= self.width or start_z >= self.height: return None
        
        end_x = min(start_x + CHUNK_SIZE + 1, self.width)
        end_z = min(start_z + CHUNK_SIZE + 1, self.height)
        
        if end_x - start_x < 2 or end_z - start_z < 2: return None
        
        chunk_list = glGenLists(1)
        glNewList(chunk_list, GL_COMPILE)
        glBegin(GL_QUADS)
        
        for z in range(start_z, end_z - 1):
            for x in range(start_x, end_x - 1):
                h00 = self.heightmap[z, x]
                h10 = self.heightmap[z, x+1]
                h01 = self.heightmap[z+1, x]
                h11 = self.heightmap[z+1, x+1]
                
                wx0 = (x - self.width/2) * TERRAIN_SCALE
                wx1 = (x + 1 - self.width/2) * TERRAIN_SCALE
                wz0 = (z - self.height/2) * TERRAIN_SCALE
                wz1 = (z + 1 - self.height/2) * TERRAIN_SCALE
                
                y00, y10, y01, y11 = h00 * HEIGHT_SCALE, h10 * HEIGHT_SCALE, h01 * HEIGHT_SCALE, h11 * HEIGHT_SCALE
                
                avg_h = (h00 + h10 + h01 + h11) / 4
                col = self.height_to_color(avg_h)
                
                dx = (h10 - h00 + h11 - h01) * HEIGHT_SCALE
                dz = (h01 - h00 + h11 - h10) * HEIGHT_SCALE
                nx, ny, nz = -dx, 2 * TERRAIN_SCALE, -dz
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0: nx, ny, nz = nx/length, ny/length, nz/length
                
                glNormal3f(nx, ny, nz)
                glColor3f(*col)
                glVertex3f(wx0, y00, wz0)
                glVertex3f(wx1, y10, wz0)
                glVertex3f(wx1, y11, wz1)
                glVertex3f(wx0, y01, wz1)
        
        glEnd()
        glEndList()
        return chunk_list
    
    def update_chunks(self, camera_chunk):
        cx, cz = camera_chunk
        needed_chunks = set()
        
        for dz in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
            for dx in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
                needed_chunks.add((cx + dx, cz + dz))
        
        to_remove = [pos for pos in self.chunks if pos not in needed_chunks]
        for pos in to_remove:
            glDeleteLists(self.chunks[pos], 1)
            del self.chunks[pos]
        
        for pos in needed_chunks:
            if pos not in self.chunks:
                chunk_list = self.create_chunk(*pos)
                if chunk_list:
                    self.chunks[pos] = chunk_list
    
    def render(self):
        for chunk_list in self.chunks.values():
            glCallList(chunk_list)
    
    def clear_all_chunks(self):
        """Clear all chunks (for when we regenerate the map)"""
        for chunk_list in self.chunks.values():
            glDeleteLists(chunk_list, 1)
        self.chunks.clear()
    
    def reload_heightmap(self, new_heightmap):
        """Reload with a new heightmap"""
        self.clear_all_chunks()
        self.heightmap = new_heightmap
        self.height, self.width = new_heightmap.shape
        self.h_min = new_heightmap.min()
        self.h_max = new_heightmap.max()
        self.chunks_x = (self.width + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.chunks_z = (self.height + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    def create_crater(self, map_x, map_z, radius):
        import random
        affected_chunks = set()
        
        for dz in range(-radius - 2, radius + 3):
            for dx in range(-radius - 2, radius + 3):
                x, z = map_x + dx, map_z + dz
                if 0 <= x < self.width and 0 <= z < self.height:
                    dist = math.sqrt(dx * dx + dz * dz)
                    if dist < radius + 1:
                        noise = random.uniform(0.6, 1.4)
                        if dist < radius * 0.3:
                            depth = 3.5 * noise
                        elif dist < radius * 0.7:
                            t = (dist - radius * 0.3) / (radius * 0.4)
                            depth = 3.5 * (1 - t * t) * noise
                        else:
                            t = (dist - radius * 0.7) / (radius * 0.3 + 1)
                            depth = 0.8 * (1 - t) * noise
                        
                        self.heightmap[z, x] -= depth
                        chunk_x = (x - self.width // 2) // CHUNK_SIZE
                        chunk_z = (z - self.height // 2) // CHUNK_SIZE
                        affected_chunks.add((chunk_x, chunk_z))
        
        for chunk_pos in affected_chunks:
            if chunk_pos in self.chunks:
                glDeleteLists(self.chunks[chunk_pos], 1)
                del self.chunks[chunk_pos]
                new_chunk = self.create_chunk(*chunk_pos)
                if new_chunk:
                    self.chunks[chunk_pos] = new_chunk


def create_water_plane():
    water_list = glGenLists(1)
    glNewList(water_list, GL_COMPILE)
    size = 2000
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glNormal3f(0, 1, 0)
    glColor4f(0.1, 0.35, 0.6, 0.7)
    glBegin(GL_QUADS)
    glVertex3f(-size, 0, -size)
    glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size)
    glVertex3f(-size, 0, size)
    glEnd()
    glDisable(GL_BLEND)
    glEndList()
    return water_list


def draw_crosshair(display):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    cx, cy = display[0] // 2, display[1] // 2
    size = 12
    
    glColor3f(0, 0, 0)
    glLineWidth(3)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size)
    glEnd()
    
    glColor3f(1, 1, 1)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size)
    glEnd()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_hud(display, camera, generator_available, seed):
    """Draw HUD with mode and info"""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Background box
    box_w = 180
    box_h = 70
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(display[0] - box_w - 10, 10)
    glVertex2f(display[0] - 10, 10)
    glVertex2f(display[0] - 10, 10 + box_h)
    glVertex2f(display[0] - box_w - 10, 10 + box_h)
    glEnd()
    
    # Mode indicator (colored bar)
    mode_y = 20
    if camera.flying:
        glColor4f(0.3, 0.6, 1.0, 0.9)  # Blue for flying
    else:
        glColor4f(0.3, 0.9, 0.3, 0.9)  # Green for walking
    
    glBegin(GL_QUADS)
    glVertex2f(display[0] - box_w, mode_y)
    glVertex2f(display[0] - 20, mode_y)
    glVertex2f(display[0] - 20, mode_y + 20)
    glVertex2f(display[0] - box_w, mode_y + 20)
    glEnd()
    
    # Mode text indicator using simple shapes
    # Draw "FLY" or "WALK" as rectangles pattern
    text_x = display[0] - box_w + 10
    text_y = mode_y + 5
    glColor4f(1, 1, 1, 1)
    
    if camera.flying:
        # F shape
        for i in range(3):
            glBegin(GL_QUADS)
            glVertex2f(text_x, text_y + i*3)
            glVertex2f(text_x + (8 if i == 0 else 5), text_y + i*3)
            glVertex2f(text_x + (8 if i == 0 else 5), text_y + i*3 + 2)
            glVertex2f(text_x, text_y + i*3 + 2)
            glEnd()
    else:
        # W shape (simplified)
        for i in range(3):
            glBegin(GL_QUADS)
            glVertex2f(text_x + i*4, text_y)
            glVertex2f(text_x + i*4 + 2, text_y)
            glVertex2f(text_x + i*4 + 2, text_y + 10)
            glVertex2f(text_x + i*4, text_y + 10)
            glEnd()
    
    # Height bar
    bar_y = 50
    glColor4f(0.5, 0.5, 0.5, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(display[0] - box_w, bar_y)
    glVertex2f(display[0] - 20, bar_y)
    glVertex2f(display[0] - 20, bar_y + 8)
    glVertex2f(display[0] - box_w, bar_y + 8)
    glEnd()
    
    # Height indicator
    height_pct = min(1, max(0, camera.y / 100))
    glColor4f(1, 0.8, 0.2, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(display[0] - box_w, bar_y)
    glVertex2f(display[0] - box_w + height_pct * (box_w - 20), bar_y)
    glVertex2f(display[0] - box_w + height_pct * (box_w - 20), bar_y + 8)
    glVertex2f(display[0] - box_w, bar_y + 8)
    glEnd()
    
    # Generator indicator
    if generator_available:
        glColor4f(0.2, 0.8, 0.2, 0.8)
        # Small "G" indicator
        glBegin(GL_QUADS)
        glVertex2f(display[0] - 35, 65)
        glVertex2f(display[0] - 20, 65)
        glVertex2f(display[0] - 20, 75)
        glVertex2f(display[0] - 35, 75)
        glEnd()
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


class Minimap:
    def __init__(self, heightmap):
        self.height, self.width = heightmap.shape
        self.texture_id = None
        self.create_texture(heightmap)
    
    def create_texture(self, heightmap):
        scale = max(1, max(self.width, self.height) // 256)
        mini_w = self.width // scale
        mini_h = self.height // scale
        
        pixels = []
        h_min, h_max = heightmap.min(), heightmap.max()
        
        for y in range(mini_h):
            for x in range(mini_w):
                src_x, src_y = x * scale, y * scale
                h = heightmap[src_y, src_x]
                
                if h < -4: r, g, b = 13, 46, 128
                elif h < -1: r, g, b = 30, 90, 160
                elif h < 0: r, g, b = 50, 120, 180
                elif h < 0.5: r, g, b = 194, 178, 128
                elif h < 1.5: r, g, b = 80, 140, 50
                elif h < 3: r, g, b = 60, 110, 40
                elif h < 4.5: r, g, b = 130, 100, 70
                else: r, g, b = 200, 200, 210
                
                pixels.extend([r, g, b])
        
        pixel_data = (GLubyte * len(pixels))(*pixels)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mini_w, mini_h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        
        self.mini_w, self.mini_h = mini_w, mini_h
        self.heightmap_width, self.heightmap_height = self.width, self.height
        self.width, self.height = heightmap.shape[1], heightmap.shape[0]
    
    def draw(self, display, camera_x, camera_z, world_width, world_height):
        map_size = 180
        margin = 10
        
        aspect = self.width / self.height
        draw_w = map_size if aspect > 1 else map_size * aspect
        draw_h = map_size / aspect if aspect > 1 else map_size
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Border
        glColor4f(0, 0, 0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(margin - 3, margin - 3)
        glVertex2f(margin + draw_w + 3, margin - 3)
        glVertex2f(margin + draw_w + 3, margin + draw_h + 3)
        glVertex2f(margin - 3, margin + draw_h + 3)
        glEnd()
        
        # Map texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1, 1, 1, 0.9)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(margin, margin)
        glTexCoord2f(1, 0); glVertex2f(margin + draw_w, margin)
        glTexCoord2f(1, 1); glVertex2f(margin + draw_w, margin + draw_h)
        glTexCoord2f(0, 1); glVertex2f(margin, margin + draw_h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        # Player marker
        norm_x = max(0, min(1, camera_x / (world_width * TERRAIN_SCALE) + 0.5))
        norm_z = max(0, min(1, camera_z / (world_height * TERRAIN_SCALE) + 0.5))
        px, py = margin + norm_x * draw_w, margin + norm_z * draw_h
        
        glColor4f(1, 0, 0, 1)
        glBegin(GL_QUADS)
        glVertex2f(px - 5, py - 5)
        glVertex2f(px + 5, py - 5)
        glVertex2f(px + 5, py + 5)
        glVertex2f(px - 5, py + 5)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 1.0, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.88, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.45, 0.45, 0.5, 1.0))
    
    glClearColor(0.55, 0.75, 0.92, 1.0)
    
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (0.55, 0.72, 0.88, 1.0))
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 80)
    glFogf(GL_FOG_END, 350)


def main():
    pygame.init()
    display = (1400, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Terrain Explorer + Generator (Press G for new map!)")
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    setup_opengl()
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(65, display[0]/display[1], 0.5, 500)
    glMatrixMode(GL_MODELVIEW)
    
    # Try to load generator
    generator = None
    if TORCH_AVAILABLE and TerrainGenerator is not None:
        try:
            generator = TerrainGenerator()
            # Test generation to catch CUDA errors early
            print("  Testing generator...")
            test_map = generator.generate(seed=1)
            print(f"  ✅ Generator works! Output shape: {test_map.shape}")
            print("  Press G to generate new maps!")
        except Exception as e:
            error_msg = str(e)
            print(f"\n  ⚠️  Generator failed!")
            print(f"  ─────────────────────────────────────────")
            
            if "CUDA" in error_msg or "cuda" in error_msg:
                print("  The model was exported with hardcoded CUDA device references.")
                print("  Your system doesn't have CUDA (NVIDIA GPU required).")
                print("")
                print("  💡 Ask Andrew to re-export with: device=context.device")
                print("     instead of: device=torch.device('cuda')")
            
            print(f"\n  Raw error:\n  {error_msg[:500]}")
            print(f"  ─────────────────────────────────────────")
            print("  Will use existing .npy files instead.\n")
            generator = None
    
    # Load initial heightmap
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy') and 'raw_map' in f]
    if npy_files:
        heights = load_heightmap(npy_files[0])
    elif generator:
        print("  Generating initial map...")
        heights = generator.generate(seed=42)
        heights = smooth_heightmap(heights)
    else:
        raise RuntimeError("No .npy heightmap found and generator not available!")
    
    chunk_manager = ChunkManager(heights)
    water_list = create_water_plane()
    minimap = Minimap(heights)
    
    camera = Camera()
    h, w = heights.shape
    camera.y = heights[h//2, w//2] * HEIGHT_SCALE + 30
    
    print("\n" + "="*60)
    print("  TERRAIN EXPLORER + GENERATOR")
    print("="*60)
    print("  MOVEMENT: WASD/Arrows | H=Up F=Down")
    print("  CAMERA: IJKL or Right-Click+Mouse")
    print("  FIRE: Space or Left-Click")
    if generator:
        print("  GENERATE: G = Generate NEW random map!")
    print("  EXIT: ESC")
    print("="*60 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    projectiles = []
    explosions = []
    last_fire_time = 0
    generation_seed = 42
    
    while running:
        dt = clock.tick(60) / 16.67
        current_time = pygame.time.get_ticks() / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if current_time - last_fire_time >= FIRE_RATE:
                        fx, fy, fz = camera.get_forward_vector()
                        projectiles.append(Projectile(
                            camera.x + fx * 2, camera.y + fy * 2, camera.z + fz * 2,
                            fx, fy, fz
                        ))
                        last_fire_time = current_time
                elif event.key == pygame.K_g and generator:
                    # Generate new map!
                    print("Generating new map...")
                    generation_seed += 1
                    new_heights = generator.generate(seed=generation_seed)
                    new_heights = smooth_heightmap(new_heights)
                    
                    # Reload everything
                    chunk_manager.reload_heightmap(new_heights)
                    heights = new_heights
                    minimap = Minimap(heights)
                    
                    # Keep camera position, but ensure we're above terrain
                    terrain_h = camera.get_terrain_height(heights, TERRAIN_SCALE, HEIGHT_SCALE)
                    min_height = terrain_h + PLAYER_HEIGHT
                    if camera.y < min_height:
                        camera.y = min_height + 5  # Place slightly above terrain
                        print(f"  (Adjusted height - was below new terrain)")
                    
                    last_chunk = None
                    
                    print(f"  New map generated! (seed: {generation_seed})")
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    mouse_look = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                elif event.button == 1:
                    if current_time - last_fire_time >= FIRE_RATE:
                        fx, fy, fz = camera.get_forward_vector()
                        projectiles.append(Projectile(
                            camera.x + fx * 2, camera.y + fy * 2, camera.z + fz * 2,
                            fx, fy, fz
                        ))
                        last_fire_time = current_time
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_look = False
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
            elif event.type == pygame.MOUSEMOTION and mouse_look:
                camera.rotate(*event.rel)
        
        keys = pygame.key.get_pressed()
        forward = right = up = 0
        speed = 2.5 if keys[pygame.K_LALT] else 1.0
        
        if keys[pygame.K_w]: forward += dt * speed
        if keys[pygame.K_s]: forward -= dt * speed
        if keys[pygame.K_a]: right -= dt * speed
        if keys[pygame.K_d]: right += dt * speed
        if keys[pygame.K_UP]: forward += dt * speed
        if keys[pygame.K_DOWN]: forward -= dt * speed
        if keys[pygame.K_LEFT]: right -= dt * speed
        if keys[pygame.K_RIGHT]: right += dt * speed
        if keys[pygame.K_h]: up += dt * speed
        if keys[pygame.K_f]: up -= dt * speed
        
        look_speed = dt * 0.8
        if keys[pygame.K_i]: camera.rotate_keyboard(0, look_speed * 1.2)
        if keys[pygame.K_k]: camera.rotate_keyboard(0, -look_speed * 1.2)
        if keys[pygame.K_j]: camera.rotate_keyboard(look_speed * 1.5, 0)
        if keys[pygame.K_l]: camera.rotate_keyboard(-look_speed * 1.5, 0)
        
        camera.move(forward, right, up, heights, TERRAIN_SCALE, HEIGHT_SCALE)
        
        current_chunk = camera.get_chunk_pos()
        if current_chunk != last_chunk:
            chunk_manager.update_chunks(current_chunk)
            last_chunk = current_chunk
        
        for proj in projectiles[:]:
            collision = proj.update(dt, heights, TERRAIN_SCALE, HEIGHT_SCALE)
            if collision:
                wx, wy, wz, map_x, map_z = collision
                explosions.append(Explosion(wx, wy, wz))
                chunk_manager.create_crater(map_x, map_z, EXPLOSION_RADIUS)
                minimap.create_texture(heights)
            if not proj.active:
                projectiles.remove(proj)
        
        for exp in explosions[:]:
            exp.update(dt)
            if not exp.active:
                explosions.remove(exp)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply()
        
        chunk_manager.render()
        glCallList(water_list)
        
        for proj in projectiles:
            proj.draw()
        for exp in explosions:
            exp.draw()
        
        draw_crosshair(display)
        draw_hud(display, camera, generator is not None, generation_seed)
        
        h, w = heights.shape
        minimap.draw(display, camera.x, camera.z, w, h)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    main()

