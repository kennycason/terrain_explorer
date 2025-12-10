"""
🌈 PSYCHEDELIC TERRAIN EXPLORER 🌈
A trippy version with breathing terrain, rainbow colors, and wild explosions!

Controls:
  - Arrow Keys or WASD: Move in facing direction
  - Hold Right Mouse: Look around
  - H/F: Fly up/down
  - Space: Fire rainbow projectiles!
  - Escape: Exit
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import GLubyte
from OpenGL.GLU import *
import numpy as np
import math
import time as time_module

# Configuration
CHUNK_SIZE = 64
CHUNK_RENDER_DISTANCE = 3
HEIGHT_SCALE = 3.5         # Reduced from 5.0 for smoother terrain
TERRAIN_SCALE = 0.8
MOVE_SPEED = 1.2
MOUSE_SENSITIVITY = 0.06

# Projectile settings
BULLET_SPEED = 3.0
BULLET_LIFETIME = 10.0
EXPLOSION_RADIUS = 15
FIRE_RATE = 0.33

# 🌈 PSYCHEDELIC SETTINGS 🌈
BREATHE_SPEED = 0.5        # How fast terrain breathes
BREATHE_AMOUNT = 0.08      # How much terrain breathes (vertical only now)
COLOR_CYCLE_SPEED = 0.3    # Rainbow cycle speed
SKY_CYCLE_SPEED = 0.1      # Sky color cycle speed

# 🦠 FRACTAL GROWTH SETTINGS 🦠
GROWTH_SPAWN_RATE = 0.02   # Chance per frame to spawn new growth
MAX_GROWTHS = 5            # Max concurrent growth colonies
GROWTH_SPEED = 0.3         # How fast growth spreads
GROWTH_STRENGTH = 0.08     # How much terrain changes per tick


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (h in 0-1 range)"""
    if s == 0:
        return (v, v, v)
    
    h = h % 1.0
    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def get_psychedelic_color(base_hue, t, saturation=0.8, value=0.9):
    """Get a cycling psychedelic color"""
    hue = (base_hue + t * COLOR_CYCLE_SPEED) % 1.0
    return hsv_to_rgb(hue, saturation, value)


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0
        self.pitch = -20
        
    def rotate(self, dx, dy):
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY
        while self.pitch > 180:
            self.pitch -= 360
        while self.pitch < -180:
            self.pitch += 360
        
    def rotate_keyboard(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch += dpitch
        while self.pitch > 180:
            self.pitch -= 360
        while self.pitch < -180:
            self.pitch += 360
        
    def move(self, forward, right, up):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        forward_y = math.sin(pitch_rad)
        forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        up_x = math.sin(yaw_rad) * math.sin(pitch_rad)
        up_y = math.cos(pitch_rad)
        up_z = math.cos(yaw_rad) * math.sin(pitch_rad)
        
        self.x += (forward * forward_x + right * right_x + up * up_x) * MOVE_SPEED
        self.y += (forward * forward_y + up * up_y) * MOVE_SPEED
        self.z += (forward * forward_z + right * right_z + up * up_z) * MOVE_SPEED
        
    def apply(self, t):
        # Removed camera wobble - was too disorienting
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    def get_chunk_pos(self):
        chunk_world_size = CHUNK_SIZE * TERRAIN_SCALE
        cx = int(self.x / chunk_world_size)
        cz = int(self.z / chunk_world_size)
        return (cx, cz)
    
    def get_forward_vector(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        fx = -math.sin(yaw_rad) * math.cos(pitch_rad)
        fy = math.sin(pitch_rad)
        fz = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        return (fx, fy, fz)


class RainbowProjectile:
    """A projectile that leaves a rainbow trail!"""
    def __init__(self, x, y, z, dx, dy, dz):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx * BULLET_SPEED
        self.dy = dy * BULLET_SPEED
        self.dz = dz * BULLET_SPEED
        self.lifetime = BULLET_LIFETIME
        self.active = True
        self.hue = np.random.random()  # Random starting color
        self.trail = []  # Trail positions
    
    def update(self, dt, heightmap, terrain_scale, height_scale):
        if not self.active:
            return None
        
        # Store trail position
        if len(self.trail) < 20:
            self.trail.append((self.x, self.y, self.z, self.hue))
        else:
            self.trail.pop(0)
            self.trail.append((self.x, self.y, self.z, self.hue))
        
        # Move projectile
        self.x += self.dx * dt
        self.y += self.dy * dt
        self.z += self.dz * dt
        self.lifetime -= dt / 60.0
        self.hue = (self.hue + 0.02) % 1.0  # Cycle color
        
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
        
        return None
    
    def draw(self, t):
        if not self.active:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Draw rainbow trail
        glBegin(GL_LINE_STRIP)
        for i, (tx, ty, tz, th) in enumerate(self.trail):
            alpha = i / len(self.trail)
            r, g, b = hsv_to_rgb(th, 1.0, 1.0)
            glColor4f(r, g, b, alpha * 0.8)
            glVertex3f(tx, ty, tz)
        glEnd()
        
        # Draw glowing sphere
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        
        r, g, b = hsv_to_rgb(self.hue, 1.0, 1.0)
        glColor4f(r, g, b, 1.0)
        
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.7 + math.sin(t * 10) * 0.2, 12, 12)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)


class PsychedelicExplosion:
    """RAINBOW EXPLOSION! 🌈💥"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.lifetime = 3.5
        self.max_lifetime = 3.5
        self.active = True
        self.base_hue = np.random.random()
    
    def update(self, dt):
        if not self.active:
            return
        self.lifetime -= dt / 60.0
        if self.lifetime <= 0:
            self.active = False
    
    def draw(self, t):
        if not self.active:
            return
        
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        size = EXPLOSION_RADIUS * TERRAIN_SCALE * (1.0 + progress * 3.0)
        alpha = 1.0 - progress * 0.7
        
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Draw multiple rainbow rings
        for i in range(8):
            ring_progress = (progress + i * 0.08) % 1.0
            ring_size = size * (0.1 + ring_progress * 0.9)
            ring_alpha = alpha * (1.0 - ring_progress * 0.8)
            
            # Rainbow colors!
            hue = (self.base_hue + i * 0.1 + t * 0.5) % 1.0
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            
            glColor4f(r, g, b, ring_alpha * 0.7)
            
            # Spiral pattern
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, ring_size * 0.3, 0)
            segments = 32
            for j in range(segments + 1):
                angle = 2 * math.pi * j / segments + t * 2
                wobble = math.sin(angle * 3 + t * 5) * ring_size * 0.1
                glVertex3f(math.cos(angle) * (ring_size + wobble), 
                          math.sin(angle * 2 + t) * ring_size * 0.5,
                          math.sin(angle) * (ring_size + wobble))
            glEnd()
        
        # Sparkle particles
        glPointSize(5)
        glBegin(GL_POINTS)
        for i in range(20):
            angle = (i / 20.0) * math.pi * 2 + t * 3
            dist = size * (0.5 + math.sin(t * 10 + i) * 0.3)
            px = math.cos(angle) * dist
            py = math.sin(t * 5 + i) * size * 0.4
            pz = math.sin(angle) * dist
            
            hue = (self.base_hue + i * 0.05 + t) % 1.0
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            glColor4f(r, g, b, alpha)
            glVertex3f(px, py, pz)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()


class FractalGrowth:
    """
    🦠 FRACTAL CANCER GROWTH 🦠
    Permanently morphs terrain in spreading fractal patterns
    """
    def __init__(self, heightmap, start_x, start_z):
        self.heightmap = heightmap
        self.height, self.width = heightmap.shape
        self.active_cells = set()  # Cells currently growing
        self.grown_cells = set()   # Cells that have finished growing
        self.active = True
        self.age = 0
        self.max_age = 500  # How long before this growth stops
        
        # Growth direction: positive = raise terrain, negative = erode
        self.direction = np.random.choice([-1, 1])
        self.strength = GROWTH_STRENGTH * np.random.uniform(0.5, 1.5)
        
        # Start from seed point
        if 0 <= start_x < self.width and 0 <= start_z < self.height:
            self.active_cells.add((start_x, start_z))
        
        # Fractal parameters
        self.branch_chance = np.random.uniform(0.1, 0.3)
        self.spread_chance = np.random.uniform(0.3, 0.6)
    
    def update(self, dt):
        """Spread the growth fractally"""
        if not self.active or not self.active_cells:
            self.active = False
            return set()
        
        self.age += dt
        if self.age > self.max_age:
            self.active = False
            return set()
        
        affected_chunks = set()
        new_active = set()
        
        for (x, z) in list(self.active_cells):
            # Modify terrain at this cell
            if 0 <= x < self.width and 0 <= z < self.height:
                # Fractal noise for organic feel
                noise = math.sin(x * 0.3) * math.cos(z * 0.3) * 0.5 + 0.5
                change = self.direction * self.strength * noise * (dt / 60.0)
                
                self.heightmap[z, x] += change
                
                # Track affected chunk
                chunk_x = (x - self.width // 2) // CHUNK_SIZE
                chunk_z = (z - self.height // 2) // CHUNK_SIZE
                affected_chunks.add((chunk_x, chunk_z))
            
            # Spread to neighbors with fractal branching
            neighbors = [
                (x+1, z), (x-1, z), (x, z+1), (x, z-1),  # Cardinals
                (x+1, z+1), (x-1, z-1), (x+1, z-1), (x-1, z+1)  # Diagonals
            ]
            
            for nx, nz in neighbors:
                if (nx, nz) not in self.grown_cells and (nx, nz) not in self.active_cells:
                    if 0 <= nx < self.width and 0 <= nz < self.height:
                        # Fractal spread probability
                        dist = math.sqrt((nx - x)**2 + (nz - z)**2)
                        spread_prob = self.spread_chance / dist
                        
                        # Add some randomness and branching
                        if np.random.random() < spread_prob:
                            if np.random.random() < self.branch_chance:
                                # Branch: start new tendril
                                new_active.add((nx, nz))
                            else:
                                new_active.add((nx, nz))
            
            # This cell is done actively growing, but stays modified
            self.grown_cells.add((x, z))
        
        # Limit spread rate
        if len(new_active) > 20:
            new_active = set(list(new_active)[:20])
        
        self.active_cells = new_active
        
        return affected_chunks


class PsychedelicChunkManager:
    """Chunk manager with TRIPPY colors!"""
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
        print(f"  🌈 PSYCHEDELIC MODE ACTIVATED! 🌈")
    
    def height_to_psychedelic_color(self, h, x, z):
        """Convert height to a psychedelic color based on position"""
        # Base hue from height
        normalized_h = (h - self.h_min) / (self.h_max - self.h_min + 0.001)
        
        # Add spatial variation
        spatial_offset = math.sin(x * 0.05) * 0.1 + math.cos(z * 0.05) * 0.1
        
        hue = (normalized_h * 0.8 + spatial_offset) % 1.0
        
        # Vary saturation and value based on height
        if h < 0:  # Water - more blue/purple tints
            saturation = 0.7
            value = 0.6 + normalized_h * 0.3
        else:  # Land - full rainbow!
            saturation = 0.85
            value = 0.7 + normalized_h * 0.3
        
        return hsv_to_rgb(hue, saturation, value)
    
    def create_chunk(self, cx, cz):
        start_x = cx * CHUNK_SIZE + self.width // 2
        start_z = cz * CHUNK_SIZE + self.height // 2
        
        if start_x < 0 or start_z < 0:
            return None
        if start_x >= self.width or start_z >= self.height:
            return None
        
        end_x = min(start_x + CHUNK_SIZE + 1, self.width)
        end_z = min(start_z + CHUNK_SIZE + 1, self.height)
        
        if end_x - start_x < 2 or end_z - start_z < 2:
            return None
        
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
                
                y00 = h00 * HEIGHT_SCALE
                y10 = h10 * HEIGHT_SCALE
                y01 = h01 * HEIGHT_SCALE
                y11 = h11 * HEIGHT_SCALE
                
                avg_h = (h00 + h10 + h01 + h11) / 4
                col = self.height_to_psychedelic_color(avg_h, x, z)
                
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
        cx, cz = camera_chunk
        needed_chunks = set()
        
        for dz in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
            for dx in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
                needed_chunks.add((cx + dx, cz + dz))
        
        to_remove = []
        for chunk_pos in self.chunks:
            if chunk_pos not in needed_chunks:
                to_remove.append(chunk_pos)
        
        for chunk_pos in to_remove:
            glDeleteLists(self.chunks[chunk_pos], 1)
            del self.chunks[chunk_pos]
        
        for chunk_pos in needed_chunks:
            if chunk_pos not in self.chunks:
                chunk_list = self.create_chunk(*chunk_pos)
                if chunk_list:
                    self.chunks[chunk_pos] = chunk_list
    
    def render(self, t):
        """Render with vertical-only breathing effect (no horizontal movement!)"""
        # Only Y-axis breathing - world doesn't move horizontally beneath you
        breathe_y = 1.0 + math.sin(t * BREATHE_SPEED) * BREATHE_AMOUNT
        
        glPushMatrix()
        # Only scale Y axis - terrain pulses up/down, not sideways
        glScalef(1.0, breathe_y, 1.0)
        
        for chunk_list in self.chunks.values():
            glCallList(chunk_list)
        
        glPopMatrix()
    
    def create_crater(self, map_x, map_z, radius):
        import random
        affected_chunks = set()
        
        for dz in range(-radius - 2, radius + 3):
            for dx in range(-radius - 2, radius + 3):
                x = map_x + dx
                z = map_z + dz
                
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
        
        return len(affected_chunks)


def create_psychedelic_water(t):
    """Create animated rainbow water"""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glNormal3f(0, 1, 0)
    
    size = 2000
    water_y = math.sin(t * 0.5) * 0.5  # Breathing water level
    
    # Rainbow water color
    hue = (t * 0.1) % 1.0
    r, g, b = hsv_to_rgb(hue, 0.6, 0.7)
    glColor4f(r * 0.3, g * 0.5 + 0.2, b * 0.8 + 0.2, 0.6)
    
    glBegin(GL_QUADS)
    glVertex3f(-size, water_y, -size)
    glVertex3f(size, water_y, -size)
    glVertex3f(size, water_y, size)
    glVertex3f(-size, water_y, size)
    glEnd()
    
    glDisable(GL_BLEND)


def draw_crosshair(display, t):
    """Rainbow crosshair!"""
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
    size = 15 + math.sin(t * 5) * 3
    
    # Rainbow crosshair
    for i in range(4):
        hue = (t * 2 + i * 0.25) % 1.0
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        glColor3f(r, g, b)
        
        glLineWidth(3)
        glBegin(GL_LINES)
        if i == 0:
            glVertex2f(cx - size, cy)
            glVertex2f(cx - 5, cy)
        elif i == 1:
            glVertex2f(cx + 5, cy)
            glVertex2f(cx + size, cy)
        elif i == 2:
            glVertex2f(cx, cy - size)
            glVertex2f(cx, cy - 5)
        else:
            glVertex2f(cx, cy + 5)
            glVertex2f(cx, cy + size)
        glEnd()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


class PsychedelicMinimap:
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
                src_x = x * scale
                src_y = y * scale
                h = heightmap[src_y, src_x]
                
                # Psychedelic minimap colors!
                normalized = (h - h_min) / (h_max - h_min + 0.001)
                hue = (normalized * 0.8 + x * 0.002 + y * 0.002) % 1.0
                
                if h < 0:
                    r, g, b = hsv_to_rgb(hue * 0.3 + 0.6, 0.7, 0.5 + normalized * 0.3)
                else:
                    r, g, b = hsv_to_rgb(hue, 0.8, 0.6 + normalized * 0.4)
                
                pixels.extend([int(r * 255), int(g * 255), int(b * 255)])
        
        pixel_data = (GLubyte * len(pixels))(*pixels)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mini_w, mini_h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        
        self.mini_w = mini_w
        self.mini_h = mini_h
    
    def draw(self, display, camera_x, camera_z, world_width, world_height, t):
        map_size = 180
        margin = 10
        map_x = margin
        map_y = margin
        
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
        
        # Rainbow border!
        hue = (t * 0.5) % 1.0
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        glColor4f(r, g, b, 0.8)
        border = 4
        glBegin(GL_QUADS)
        glVertex2f(map_x - border, map_y - border)
        glVertex2f(map_x + draw_w + border, map_y - border)
        glVertex2f(map_x + draw_w + border, map_y + draw_h + border)
        glVertex2f(map_x - border, map_y + draw_h + border)
        glEnd()
        
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
        
        norm_x = (camera_x / (world_width * TERRAIN_SCALE) + 0.5)
        norm_z = (camera_z / (world_height * TERRAIN_SCALE) + 0.5)
        norm_x = max(0, min(1, norm_x))
        norm_z = max(0, min(1, norm_z))
        
        player_map_x = map_x + norm_x * draw_w
        player_map_y = map_y + norm_z * draw_h
        
        # Pulsing rainbow player marker
        marker_size = 5 + math.sin(t * 5) * 2
        hue = (t * 2) % 1.0
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        glColor4f(r, g, b, 1)
        glBegin(GL_QUADS)
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


def setup_psychedelic_opengl(t):
    """Setup with cycling colors!"""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Cycling light color!
    light_hue = (t * 0.2) % 1.0
    lr, lg, lb = hsv_to_rgb(light_hue, 0.3, 1.0)
    
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 1.0, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (lr, lg, lb, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.5, 1.0))
    
    # Cycling sky color!
    sky_hue = (t * SKY_CYCLE_SPEED) % 1.0
    sr, sg, sb = hsv_to_rgb(sky_hue, 0.4, 0.85)
    glClearColor(sr, sg, sb, 1.0)
    
    # Cycling fog!
    fog_hue = (t * SKY_CYCLE_SPEED + 0.1) % 1.0
    fr, fg, fb = hsv_to_rgb(fog_hue, 0.3, 0.9)
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (fr, fg, fb, 1.0))
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 80)
    glFogf(GL_FOG_END, 350)


def smooth_heightmap(data, iterations=2, kernel_size=3):
    """Apply smoothing to reduce spikiness"""
    from scipy.ndimage import uniform_filter
    
    smoothed = data.copy()
    for _ in range(iterations):
        smoothed = uniform_filter(smoothed, size=kernel_size, mode='nearest')
    
    return smoothed


def load_heightmap(filename, smooth=True):
    print("Loading heightmap...")
    data = np.load(filename)
    
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    
    print(f"  Shape: {data.shape}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")
    
    if smooth:
        print("  Applying smoothing to reduce spikiness...")
        data = smooth_heightmap(data, iterations=2, kernel_size=3)
        print(f"  Smoothed range: {data.min():.2f} to {data.max():.2f}")
    
    return data.astype(np.float32)


def main():
    pygame.init()
    display = (1400, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("🌈 PSYCHEDELIC TERRAIN EXPLORER 🌈")
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(65, display[0]/display[1], 0.5, 500)
    glMatrixMode(GL_MODELVIEW)
    
    heights = load_heightmap('raw_map_20251210_134411.npy')
    
    chunk_manager = PsychedelicChunkManager(heights)
    minimap = PsychedelicMinimap(heights)
    
    camera = Camera()
    h, w = heights.shape
    center_height = heights[h//2, w//2]
    camera.y = center_height * HEIGHT_SCALE + 30
    
    print("\n" + "🌈"*30)
    print("  PSYCHEDELIC TERRAIN EXPLORER")
    print("🌈"*30)
    print("  MOVEMENT: WASD/Arrows | H=Up F=Down")
    print("  CAMERA: IJKL or Right-Click+Mouse")
    print("  FIRE: Space or Left-Click")
    print("  EXIT: ESC")
    print()
    print("  ✨ ENJOY THE TRIP! ✨")
    print("🌈"*30 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    
    projectiles = []
    explosions = []
    growths = []  # 🦠 Fractal growth colonies
    last_fire_time = 0
    start_time = time_module.time()
    growth_update_timer = 0
    
    while running:
        dt = clock.tick(60) / 16.67
        t = time_module.time() - start_time  # Time for animations
        current_time = pygame.time.get_ticks() / 1000.0
        
        # Update OpenGL settings with time for color cycling
        setup_psychedelic_opengl(t)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if current_time - last_fire_time >= FIRE_RATE:
                        fx, fy, fz = camera.get_forward_vector()
                        bullet = RainbowProjectile(
                            camera.x + fx * 2,
                            camera.y + fy * 2,
                            camera.z + fz * 2,
                            fx, fy, fz
                        )
                        projectiles.append(bullet)
                        last_fire_time = current_time
                        
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    mouse_look = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                elif event.button == 1:
                    if current_time - last_fire_time >= FIRE_RATE:
                        fx, fy, fz = camera.get_forward_vector()
                        bullet = RainbowProjectile(
                            camera.x + fx * 2,
                            camera.y + fy * 2,
                            camera.z + fz * 2,
                            fx, fy, fz
                        )
                        projectiles.append(bullet)
                        last_fire_time = current_time
                        
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
        if keys[pygame.K_q]: camera.rotate_keyboard(look_speed * 1.5, 0)
        if keys[pygame.K_e]: camera.rotate_keyboard(-look_speed * 1.5, 0)
        if keys[pygame.K_r]: camera.rotate_keyboard(0, look_speed * 1.2)
        
        camera.move(forward, right, up)
        
        current_chunk = camera.get_chunk_pos()
        if current_chunk != last_chunk:
            chunk_manager.update_chunks(current_chunk)
            last_chunk = current_chunk
        
        for proj in projectiles[:]:
            collision = proj.update(dt, heights, TERRAIN_SCALE, HEIGHT_SCALE)
            if collision:
                wx, wy, wz, map_x, map_z = collision
                explosions.append(PsychedelicExplosion(wx, wy, wz))
                chunk_manager.create_crater(map_x, map_z, EXPLOSION_RADIUS)
                minimap.create_texture(heights)
            
            if not proj.active:
                projectiles.remove(proj)
        
        for exp in explosions[:]:
            exp.update(dt)
            if not exp.active:
                explosions.remove(exp)
        
        # 🦠 Update fractal growths
        growth_update_timer += dt
        if growth_update_timer > 5:  # Update growths every ~5 frames for performance
            growth_update_timer = 0
            
            # Chance to spawn new growth
            if len(growths) < MAX_GROWTHS and np.random.random() < GROWTH_SPAWN_RATE:
                # Spawn at random location on terrain
                spawn_x = np.random.randint(0, w)
                spawn_z = np.random.randint(0, h)
                growths.append(FractalGrowth(heights, spawn_x, spawn_z))
            
            # Update existing growths
            all_affected_chunks = set()
            for growth in growths[:]:
                affected = growth.update(dt)
                all_affected_chunks.update(affected)
                
                if not growth.active:
                    growths.remove(growth)
            
            # Regenerate affected chunks
            if all_affected_chunks:
                for chunk_pos in all_affected_chunks:
                    if chunk_pos in chunk_manager.chunks:
                        glDeleteLists(chunk_manager.chunks[chunk_pos], 1)
                        del chunk_manager.chunks[chunk_pos]
                        new_chunk = chunk_manager.create_chunk(*chunk_pos)
                        if new_chunk:
                            chunk_manager.chunks[chunk_pos] = new_chunk
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply(t)
        
        chunk_manager.render(t)
        create_psychedelic_water(t)
        
        for proj in projectiles:
            proj.draw(t)
        
        for exp in explosions:
            exp.draw(t)
        
        draw_crosshair(display, t)
        
        h, w = heights.shape
        minimap.draw(display, camera.x, camera.z, w, h, t)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    main()
