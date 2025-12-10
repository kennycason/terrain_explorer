"""
3D Terrain Explorer with Dynamic Chunk Loading
Uses the full resolution .npy heightmap and loads chunks as you explore

Controls:
  - Arrow Keys or WASD: Move in facing direction
  - Hold Right Mouse: Look around
  - Space: Fly up | C/X/Z: Fly down
  - Q/E: Rotate camera | R/F: Look up/down
  - Escape: Exit
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import GLubyte
from OpenGL.GLU import *
import numpy as np
import math

# Configuration
CHUNK_SIZE = 64          # Size of each chunk in grid cells
CHUNK_RENDER_DISTANCE = 3  # How many chunks to render around player
HEIGHT_SCALE = 5.0       # Vertical scale (adjusted for smoother terrain)
TERRAIN_SCALE = 0.8      # Horizontal scale
MOVE_SPEED = 1.2
MOUSE_SENSITIVITY = 0.06

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0      # Horizontal rotation (left/right)
        self.pitch = -20  # Vertical rotation (up/down)
        
    def rotate(self, dx, dy):
        """Mouse rotation - allows full loops!"""
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY
        # Allow full 360° loops - normalize to -180 to 180
        while self.pitch > 180:
            self.pitch -= 360
        while self.pitch < -180:
            self.pitch += 360
        
    def rotate_keyboard(self, dyaw, dpitch):
        """Keyboard rotation (Q/E/R/F or JKLI) - allows full loops!"""
        self.yaw += dyaw
        self.pitch += dpitch
        # Allow full 360° loops
        while self.pitch > 180:
            self.pitch -= 360
        while self.pitch < -180:
            self.pitch += 360
        
    def move(self, forward, right, up):
        """
        Flight-style movement (allows loops and rolls):
        - Forward moves in the FULL 3D direction we're facing (including pitch)
        - Right strafes perpendicular
        - Up is relative to our orientation
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        # Forward vector - full 3D direction we're looking
        # Accounts for both yaw AND pitch for proper flight
        forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        forward_y = math.sin(pitch_rad)
        forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Right vector - perpendicular to forward, stays horizontal
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Up vector - perpendicular to both forward and right
        up_x = math.sin(yaw_rad) * math.sin(pitch_rad)
        up_y = math.cos(pitch_rad)
        up_z = math.cos(yaw_rad) * math.sin(pitch_rad)
        
        # Apply movement
        self.x += (forward * forward_x + right * right_x + up * up_x) * MOVE_SPEED
        self.y += (forward * forward_y + up * up_y) * MOVE_SPEED
        self.z += (forward * forward_z + right * right_z + up * up_z) * MOVE_SPEED
        
    def apply(self):
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    def get_chunk_pos(self):
        """Get which chunk the camera is in"""
        chunk_world_size = CHUNK_SIZE * TERRAIN_SCALE
        cx = int(self.x / chunk_world_size)
        cz = int(self.z / chunk_world_size)
        return (cx, cz)


class ChunkManager:
    def __init__(self, heightmap):
        self.heightmap = heightmap
        self.height, self.width = heightmap.shape
        self.chunks = {}  # (cx, cz) -> display_list
        self.h_min = heightmap.min()
        self.h_max = heightmap.max()
        
        # Calculate how many chunks we have
        self.chunks_x = (self.width + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.chunks_z = (self.height + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        print(f"  World size: {self.width}x{self.height}")
        print(f"  Chunk grid: {self.chunks_x}x{self.chunks_z} chunks")
        print(f"  Height range: {self.h_min:.2f} to {self.h_max:.2f}")
    
    def height_to_color(self, h):
        """Convert height to terrain color"""
        if h < -4:
            return (0.05, 0.18, 0.5)
        elif h < -1:
            t = (h + 4) / 3
            return (0.08 + t*0.12, 0.28 + t*0.17, 0.55 + t*0.1)
        elif h < 0:
            return (0.18, 0.48, 0.68)
        elif h < 0.3:
            return (0.78, 0.72, 0.52)
        elif h < 1.0:
            t = (h - 0.3) / 0.7
            return (0.28 + t*0.02, 0.58 - t*0.08, 0.22)
        elif h < 2.0:
            return (0.22, 0.48, 0.18)
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
        """Create a display list for a specific chunk"""
        # Calculate pixel coordinates for this chunk
        start_x = cx * CHUNK_SIZE + self.width // 2
        start_z = cz * CHUNK_SIZE + self.height // 2
        
        # Check bounds
        if start_x < 0 or start_z < 0:
            return None
        if start_x >= self.width or start_z >= self.height:
            return None
        
        end_x = min(start_x + CHUNK_SIZE + 1, self.width)
        end_z = min(start_z + CHUNK_SIZE + 1, self.height)
        
        if end_x - start_x < 2 or end_z - start_z < 2:
            return None
        
        # Create display list
        chunk_list = glGenLists(1)
        glNewList(chunk_list, GL_COMPILE)
        glBegin(GL_QUADS)
        
        for z in range(start_z, end_z - 1):
            for x in range(start_x, end_x - 1):
                # Get heights
                h00 = self.heightmap[z, x]
                h10 = self.heightmap[z, x+1]
                h01 = self.heightmap[z+1, x]
                h11 = self.heightmap[z+1, x+1]
                
                # World positions (centered on origin)
                wx0 = (x - self.width/2) * TERRAIN_SCALE
                wx1 = (x + 1 - self.width/2) * TERRAIN_SCALE
                wz0 = (z - self.height/2) * TERRAIN_SCALE
                wz1 = (z + 1 - self.height/2) * TERRAIN_SCALE
                
                y00 = h00 * HEIGHT_SCALE
                y10 = h10 * HEIGHT_SCALE
                y01 = h01 * HEIGHT_SCALE
                y11 = h11 * HEIGHT_SCALE
                
                # Color based on average height
                avg_h = (h00 + h10 + h01 + h11) / 4
                col = self.height_to_color(avg_h)
                
                # Normal
                dx = (h10 - h00 + h11 - h01) * HEIGHT_SCALE
                dz = (h01 - h00 + h11 - h10) * HEIGHT_SCALE
                nx, ny, nz = -dx, 2 * TERRAIN_SCALE, -dz
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    nx, ny, nz = nx/length, ny/length, nz/length
                
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
        """Load/unload chunks based on camera position"""
        cx, cz = camera_chunk
        needed_chunks = set()
        
        # Determine which chunks we need
        for dz in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
            for dx in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
                needed_chunks.add((cx + dx, cz + dz))
        
        # Unload chunks that are too far
        to_remove = []
        for chunk_pos in self.chunks:
            if chunk_pos not in needed_chunks:
                to_remove.append(chunk_pos)
        
        for chunk_pos in to_remove:
            glDeleteLists(self.chunks[chunk_pos], 1)
            del self.chunks[chunk_pos]
        
        # Load new chunks
        for chunk_pos in needed_chunks:
            if chunk_pos not in self.chunks:
                chunk_list = self.create_chunk(*chunk_pos)
                if chunk_list:
                    self.chunks[chunk_pos] = chunk_list
    
    def render(self):
        """Render all loaded chunks"""
        for chunk_list in self.chunks.values():
            glCallList(chunk_list)


def create_water_plane():
    """Create a large water plane"""
    water_list = glGenLists(1)
    glNewList(water_list, GL_COMPILE)
    
    size = 2000  # Large water plane
    water_y = 0.0
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glNormal3f(0, 1, 0)
    glColor4f(0.1, 0.35, 0.6, 0.7)
    
    glBegin(GL_QUADS)
    glVertex3f(-size, water_y, -size)
    glVertex3f(size, water_y, -size)
    glVertex3f(size, water_y, size)
    glVertex3f(-size, water_y, size)
    glEnd()
    
    glDisable(GL_BLEND)
    glEndList()
    
    return water_list


def draw_crosshair(display):
    """Draw crosshair"""
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
    glVertex2f(cx - size, cy)
    glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size)
    glVertex2f(cx, cy + size)
    glEnd()
    
    glColor3f(1, 1, 1)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy)
    glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size)
    glVertex2f(cx, cy + size)
    glEnd()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


class Minimap:
    def __init__(self, heightmap):
        """Create a minimap texture from the heightmap"""
        self.height, self.width = heightmap.shape
        self.texture_id = None
        self.create_texture(heightmap)
    
    def create_texture(self, heightmap):
        """Generate an OpenGL texture from heightmap data"""
        # Downsample for minimap (max 256x256)
        scale = max(1, max(self.width, self.height) // 256)
        mini_w = self.width // scale
        mini_h = self.height // scale
        
        # Create RGB image data
        pixels = []
        h_min, h_max = heightmap.min(), heightmap.max()
        
        for y in range(mini_h):
            for x in range(mini_w):
                # Sample from original heightmap
                src_x = x * scale
                src_y = y * scale
                h = heightmap[src_y, src_x]
                
                # Color based on height
                if h < -4:
                    r, g, b = 13, 46, 128      # Deep ocean
                elif h < -1:
                    r, g, b = 30, 90, 160      # Ocean
                elif h < 0:
                    r, g, b = 50, 120, 180     # Shallow
                elif h < 0.5:
                    r, g, b = 194, 178, 128    # Beach
                elif h < 1.5:
                    r, g, b = 80, 140, 50      # Grass
                elif h < 3:
                    r, g, b = 60, 110, 40      # Forest
                elif h < 4.5:
                    r, g, b = 130, 100, 70     # Highland
                else:
                    r, g, b = 200, 200, 210    # Snow
                
                pixels.extend([r, g, b])
        
        # Create OpenGL texture
        pixel_data = (GLubyte * len(pixels))(*pixels)
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mini_w, mini_h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        
        self.mini_w = mini_w
        self.mini_h = mini_h
    
    def draw(self, display, camera_x, camera_z, world_width, world_height):
        """Draw the minimap with player position"""
        # Minimap size and position
        map_size = 180
        margin = 10
        map_x = margin
        map_y = margin
        
        # Calculate aspect ratio
        aspect = self.width / self.height
        if aspect > 1:
            draw_w = map_size
            draw_h = map_size / aspect
        else:
            draw_h = map_size
            draw_w = map_size * aspect
        
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
        
        # Draw border/background
        glColor4f(0, 0, 0, 0.7)
        border = 3
        glBegin(GL_QUADS)
        glVertex2f(map_x - border, map_y - border)
        glVertex2f(map_x + draw_w + border, map_y - border)
        glVertex2f(map_x + draw_w + border, map_y + draw_h + border)
        glVertex2f(map_x - border, map_y + draw_h + border)
        glEnd()
        
        # Draw minimap texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1, 1, 1, 0.9)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(map_x, map_y)
        glTexCoord2f(1, 0); glVertex2f(map_x + draw_w, map_y)
        glTexCoord2f(1, 1); glVertex2f(map_x + draw_w, map_y + draw_h)
        glTexCoord2f(0, 1); glVertex2f(map_x, map_y + draw_h)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        
        # Draw player position as red square
        # Convert world position to minimap position
        # World center is at (0,0), map goes from -width/2 to +width/2
        norm_x = (camera_x / (world_width * TERRAIN_SCALE) + 0.5)
        norm_z = (camera_z / (world_height * TERRAIN_SCALE) + 0.5)
        
        # Clamp to map bounds
        norm_x = max(0, min(1, norm_x))
        norm_z = max(0, min(1, norm_z))
        
        player_map_x = map_x + norm_x * draw_w
        player_map_y = map_y + norm_z * draw_h
        
        # Draw red player marker
        marker_size = 5
        glColor4f(1, 0, 0, 1)
        glBegin(GL_QUADS)
        glVertex2f(player_map_x - marker_size, player_map_y - marker_size)
        glVertex2f(player_map_x + marker_size, player_map_y - marker_size)
        glVertex2f(player_map_x + marker_size, player_map_y + marker_size)
        glVertex2f(player_map_x - marker_size, player_map_y + marker_size)
        glEnd()
        
        # Draw red border around marker
        glColor4f(0.5, 0, 0, 1)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(player_map_x - marker_size, player_map_y - marker_size)
        glVertex2f(player_map_x + marker_size, player_map_y - marker_size)
        glVertex2f(player_map_x + marker_size, player_map_y + marker_size)
        glVertex2f(player_map_x - marker_size, player_map_y + marker_size)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def setup_opengl():
    """Configure OpenGL"""
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


def load_heightmap(filename):
    """Load the raw .npy height map at full resolution"""
    print("Loading heightmap...")
    data = np.load(filename)
    
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    
    print(f"  Shape: {data.shape}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")
    
    return data.astype(np.float32)


def main():
    pygame.init()
    display = (1400, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Terrain Explorer - Dynamic Chunk Loading")
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    setup_opengl()
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(65, display[0]/display[1], 0.5, 500)
    glMatrixMode(GL_MODELVIEW)
    
    # Load heightmap at full resolution
    heights = load_heightmap('raw_map_20251210_134411.npy')
    
    # Create chunk manager and minimap
    chunk_manager = ChunkManager(heights)
    water_list = create_water_plane()
    minimap = Minimap(heights)
    
    # Camera - start at center
    camera = Camera()
    h, w = heights.shape
    center_height = heights[h//2, w//2]
    camera.y = center_height * HEIGHT_SCALE + 30
    
    print("\n" + "="*60)
    print("  TERRAIN EXPLORER - JET FLIGHT MODE")
    print("="*60)
    print("  LEFT STICK (Movement):")
    print("    W / ↑    - Fly forward (in facing direction)")
    print("    S / ↓    - Fly backward")
    print("    A / ←    - Strafe left")
    print("    D / →    - Strafe right")
    print("    Space    - Fly UP (relative to orientation)")
    print("    C/X/Z    - Fly DOWN")
    print()
    print("  RIGHT STICK (Camera):")
    print("    I        - Pitch UP (do loops!)")
    print("    K        - Pitch DOWN")
    print("    J        - Look LEFT")
    print("    L        - Look RIGHT")
    print("    (or Q/E/R/F as alternative)")
    print()
    print("  Mouse: Hold Right-Click to look around")
    print("  Alt: Move faster | ESC: Exit")
    print()
    print("  TIP: Hold I to do a full loop like a jet!")
    print("  Minimap in top-left shows your position (red square)")
    print("="*60 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    
    while running:
        dt = clock.tick(60) / 16.67
        fps = clock.get_fps()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    mouse_look = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_look = False
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
            elif event.type == pygame.MOUSEMOTION and mouse_look:
                dx, dy = event.rel
                camera.rotate(dx, dy)
        
        keys = pygame.key.get_pressed()
        forward = right = up = 0
        speed = 2.5 if keys[pygame.K_LALT] else 1.0
        
        # === LEFT STICK: Movement (WASD or Arrow Keys) ===
        # WASD
        if keys[pygame.K_w]: forward += dt * speed
        if keys[pygame.K_s]: forward -= dt * speed
        if keys[pygame.K_a]: right -= dt * speed
        if keys[pygame.K_d]: right += dt * speed
        
        # Arrow keys (same as WASD)
        if keys[pygame.K_UP]: forward += dt * speed
        if keys[pygame.K_DOWN]: forward -= dt * speed
        if keys[pygame.K_LEFT]: right -= dt * speed
        if keys[pygame.K_RIGHT]: right += dt * speed
        
        # Up/Down flight
        if keys[pygame.K_SPACE]: up += dt * speed
        if keys[pygame.K_c] or keys[pygame.K_x] or keys[pygame.K_z]: up -= dt * speed
        
        # === RIGHT STICK: Camera Look (IJKL or Q/E/R/F) ===
        look_speed = dt * 0.8  # Smooth control
        
        # IJKL - like right joystick
        # I = look up, K = look down, J = look left, L = look right
        if keys[pygame.K_i]: camera.rotate_keyboard(0, look_speed * 1.2)    # Look UP
        if keys[pygame.K_k]: camera.rotate_keyboard(0, -look_speed * 1.2)   # Look DOWN
        if keys[pygame.K_j]: camera.rotate_keyboard(look_speed * 1.5, 0)    # Look LEFT (+ yaw = left)
        if keys[pygame.K_l]: camera.rotate_keyboard(-look_speed * 1.5, 0)   # Look RIGHT (- yaw = right)
        
        # Q/E/R/F - alternative camera controls
        if keys[pygame.K_q]: camera.rotate_keyboard(look_speed * 1.5, 0)    # Look LEFT
        if keys[pygame.K_e]: camera.rotate_keyboard(-look_speed * 1.5, 0)   # Look RIGHT
        if keys[pygame.K_r]: camera.rotate_keyboard(0, look_speed * 1.2)    # Look UP
        if keys[pygame.K_f]: camera.rotate_keyboard(0, -look_speed * 1.2)   # Look DOWN
        
        camera.move(forward, right, up)
        
        # Update chunks when camera moves to a new chunk
        current_chunk = camera.get_chunk_pos()
        if current_chunk != last_chunk:
            chunk_manager.update_chunks(current_chunk)
            last_chunk = current_chunk
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply()
        
        chunk_manager.render()
        glCallList(water_list)
        draw_crosshair(display)
        
        # Draw minimap with player position
        h, w = heights.shape
        minimap.draw(display, camera.x, camera.z, w, h)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    main()
