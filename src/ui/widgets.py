# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalContextManager=false
# pyright: reportOptionalMemberAccess=false
import random

import numpy as np
from typing import Tuple

# import pymunk
from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import Mesh, PopMatrix, PushMatrix
from kivy.logger import Logger
from kivy.properties import (
    ColorProperty,
    ListProperty,
    NumericProperty,
)
from kivy.uix.behaviors import TouchRippleBehavior
from kivy.uix.label import Label
from kivy.uix.label import Label as PopupLabel
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from numpy.typing import NDArray

from engine.components.gravity import update_gravity

# from kivy.uix.boxlayout import BoxLayout
# from scipy.signal import convolve2d


class BackgroundColor:
    background_color: ListProperty = ListProperty([1, 1, 1, 1])


class PickerLabel(Label):
    background_color: ColorProperty = ColorProperty([0.9, 0.9, 0.9, 1])
    r: NumericProperty = NumericProperty(0)

    def __init__(self, **kwargs):
        super(PickerLabel, self).__init__(**kwargs)
        self.bind()

    def _update_index(self, instance, value):
        column = self.parent
        if column:
            column.remove_widget(self)
            column.add_widget(self, index=value)

    def on_color_touch(self, touch):
        if self.collide_point(*touch.pos) and not self.text:
            Logger.info(f"Clicked color: {self.background_color}")
            popup: Popup = Popup(
                title="Color Value",
                content=PopupLabel(text=f"RGB: {self.background_color}"),
                size_hint=(0.3, 0.3),
            )
            popup.open()


Factory.register("PickerLabel", cls=PickerLabel)


class RippleButton(TouchRippleBehavior, Label):
    def __init__(self, **kwargs):
        super(RippleButton, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Logger.debug(f"Touch down at: {touch.pos}")
            self.ripple_show(touch)
            return True
        return super(RippleButton, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            # Logger.debug(f"Touch up at: {touch.pos}")
            self.ripple_fade()
            self.app = App.get_running_app()
            if self.app is None:
                return
            self.app.open_modal()
            return True
        return super(RippleButton, self).on_touch_up(touch)


STATES = {"solid": 0, "liquid": 1, "gas": 2}


class SimulationGrid(Widget):
    def __init__(self, pixel_size: int = 4, **kwargs):
        super().__init__(**kwargs)

        self._edge_mode: int = 0
        self.pixel_size: int = pixel_size
        self.on_resized = None
        self.is_initialized: bool = False
        self.is_ready: bool = False

        # Element definitions (map element names to IDs)
        self.element_map: dict = {}
        self.element_properties: dict = {}
        self.particle_texture = None
        self.texture_data = None
        self.mesh: Mesh | None = None
        self._render_arrays_allocated: bool = False
        self._max_allocated_particles: int = 0
        # self._temp_positions = None
        # self._temp_colors = None
        self._temp_indices = None
        self._dirty_texture: bool = False

        self._init_elements()
        Clock.schedule_once(self._check_initialization, 0.1)
        self._color_map = self._build_material_colors()

    def _build_material_colors(self):
        """Build a safe, oversized color lookup table"""
        max_id = 1000

        # Create lookup: index 0 = empty, 1-N = materials
        colors: NDArray = np.zeros((max_id + 1, 4), dtype=np.uint8)
        colors[0] = [0, 0, 0, 0]  # transparent (for -1 → 0)

        # Fill known materials
        idx: int = 0
        for name, elem_id in self.element_map.items():
            if elem_id is None:
                # Auto-assign if missing
                elem_id = idx
                self.element_map[name] = elem_id
            idx += 1
            props = self.element_properties[elem_id][1]
            color = props.get("intrinsic_properties", {}).get(
                "color", [1.0, 1.0, 0.0, 1.0]
            )
            rgba = np.array(color[:4], dtype=np.float32)
            rgba[:4] *= 255
            # rgba[3] *= 255
            colors[elem_id + 1] = rgba.astype(np.uint8)

        Logger.info(f"Built color map with {len(colors)} entries (max ID: {max_id})")
        return colors

    def switch_edge_mode(self, new_mode):
        self._edge_mode = new_mode
        print(f"Switching edge mode → {new_mode}")

        # Force rebuild to eliminate ghost grid cells
        self.rebuild_spatial_grid()

        # Ensure all particles are written to the grid at their current buffer positions
        buffer = self.buffers[0]
        particles = buffer["particles"]
        spatial_grid = buffer["spatial_grid"]
        count = buffer["particle_count"]

        for i in range(count):
            if not particles["active"][i]:
                continue
            x = int(particles["x"][i])
            y = int(particles["y"][i])
            # x = particles["x"][i]
            # y = particles["y"][i]
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                spatial_grid[x, y] = i

        # Apply velocity nudge to active particles
        active_mask = np.zeros(len(particles["active"]), dtype=np.bool_)
        active_mask[:count] = particles["active"][:count]
        particles["velocity_y"][active_mask] -= 0.5  # Nudge downward

        # Flag so gravity update can skip damping one frame
        self.just_switched_edge_mode = True

    def _allocate_render_arrays(self, max_particles):
        """Allocate rendering arrays once and reuse them"""
        if max_particles <= self._max_allocated_particles:
            return  # Already have enough space

        # Only reallocate if we need significantly more space (avoid frequent reallocations)
        new_size = max(max_particles, int(self._max_allocated_particles * 1.5))

        self._temp_positions = np.empty((new_size, 2), dtype=np.int32)
        self._temp_colors = np.empty((new_size, 4), dtype=np.uint8)
        self._temp_indices = np.empty(new_size, dtype=np.int32)
        self._max_allocated_particles = new_size
        self._render_arrays_allocated = True

    def _init_elements(self):
        """Initialize element mappings"""
        app = App.get_running_app()
        if app:
            self.app = app
        if not hasattr(self.app, "elements"):
            raise ValueError("App does not have 'elements' attribute")

        for i, (name, props) in enumerate(self.app.elements.items()):
            self.element_map[name] = i
            self.element_properties[i] = [name, props]

    def _check_initialization(self, dt):
        """Check if we're ready to initialize the simulation"""
        if self.size == [1, 1] or self.size == [100, 100]:
            # Still using default size, wait longer
            Clock.schedule_once(self._check_initialization, 0.1)
            return

        if not self.is_initialized:
            self._initialize_simulation()

    def _initialize_simulation(self):
        """Initialize the simulation with final size"""
        if self.is_initialized:
            return

        Logger.debug(f"Initializing simulation with size: {self.size}")

        self.grid_width = max(1, int(self.width / self.pixel_size))
        self.grid_height = max(1, int(self.height / self.pixel_size))

        Logger.debug(f"Grid dimensions: {self.grid_width}x{self.grid_height}")

        max_particles = self.grid_width * self.grid_height
        self.buffers = [
            self._create_buffer(max_particles),
            self._create_buffer(max_particles),
        ]

        self._setup_texture_rendering(self.grid_width, self.grid_height)

        self.is_initialized = True

        self.is_ready = True

        # Clock.schedule_interval(lambda dt: self.debug_spatial_grid(limit=20), 3.0)

        Logger.debug("Simulation initialization complete")

    def wait_for_ready(self, timeout=5.0):
        """Block until simulation is ready or timeout"""
        import time

        start_time = time.time()

        while not self.is_ready and (time.time() - start_time) < timeout:
            # Process Kivy events to allow size updates
            from kivy.base import EventLoop

            EventLoop.idle()
            time.sleep(0.01)

        return self.is_ready

    def _create_buffer(self, max_particles):
        dtype = [
            ("element_id", np.uint16),
            ("x", np.float32),
            ("y", np.float32),
            ("temperature", np.float32),
            ("mass", np.float32),
            ("density", np.float32),
            ("drag_coeff", np.float32),
            ("state", np.int32),
            ("velocity_x", np.float32),
            ("velocity_y", np.float32),
            ("active", bool),
            ("burning", bool),
            ("melting", bool),
            ("color", np.float32, 4),
        ]

        fall_dtype = np.dtype(
            [
                ("fall_old_x", np.int32),
                ("fall_old_y", np.int32),
                ("fall_new_x", np.int32),
                ("fall_new_y", np.int32),
                ("fall_mask", np.bool_),
            ]
        )

        slide_dtype = np.dtype(
            [
                ("slide_old_x", np.int32),
                ("slide_old_y", np.int32),
                ("slide_new_x", np.int32),
                ("slide_new_y", np.int32),
                ("slide_mask", np.bool_),
            ]
        )

        return {
            "particles": np.zeros(max_particles, dtype=dtype),
            "fall_movement": np.zeros(max_particles, dtype=fall_dtype),
            "slide_movement": np.zeros(max_particles, dtype=slide_dtype),
            "spatial_grid": np.full(
                (self.grid_width, self.grid_height), -1, dtype=np.int32
            ),
            "particle_count": 0,
            "_burning_particles": set(),
            "_conductive_particles": set(),
            "_reactive_particles": set(),
        }

    def add_particle(self, x, y, element_name, color=None):
        """Add a particle at position (x, y)"""
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        if (
            self.buffers[0]["particle_count"]
            >= self.buffers[0]["particles"]["element_id"].size
        ):
            return False  # Grid full
        if self.buffers[0]["spatial_grid"][x, y] != -1:
            return False  # Position occupied

        # Get element ID
        element_id = self.element_map.get(element_name, 0)

        # Add particle data
        idx = self.buffers[0]["particle_count"]
        self.buffers[0]["particles"]["element_id"][idx] = element_id
        self.buffers[0]["particles"]["x"][idx] = x
        self.buffers[0]["particles"]["y"][idx] = y
        self.buffers[0]["particles"]["active"][idx] = True

        # Set properties from element definition
        if element_id in self.element_properties:
            props = self.element_properties[element_id][1]
            intrinsic = props.get("intrinsic_properties", {})
            dynamic = props.get("dynamic_properties", {})
            self.buffers[0]["particles"]["drag_coeff"][idx] = intrinsic.get(
                "drag_coeff", None
            )
            self.buffers[0]["particles"]["mass"][idx] = intrinsic.get("mass", 1.0)
            self.buffers[0]["particles"]["temperature"][idx] = dynamic.get(
                "temperature", 20.0
            )
            self.buffers[0]["particles"]["state"][idx] = STATES.get(
                intrinsic.get("state", "solid"), 0
            )  # 0=solid,1=liquid,2=gas

            # Set color
            if color:
                self.buffers[0]["particles"]["color"][idx] = (
                    color[:4] if len(color) >= 4 else [*color, 1.0]
                )
            else:
                element_color = intrinsic.get("color", [1.0, 1.0, 1.0, 1.0])
                self.buffers[0]["particles"]["color"][idx] = (
                    element_color[:4]
                    if len(element_color) >= 4
                    else [*element_color, 1.0]
                )
            # self.pymunk_space.add(pymunk.Circle(radius=4))

        self.buffers[0]["particles"]["velocity_x"][idx] = random.uniform(-0.1, 0.1)
        self.buffers[0]["particles"]["velocity_y"][idx] = random.uniform(
            -0.5, -0.1
        )  # Small downward velocity

        # Update spatial grid
        self.buffers[0]["spatial_grid"][x, y] = idx
        self.buffers[0]["particle_count"] += 1

        if not self._dirty_texture:
            self.mark_particles_dirty()

        return True

    def remove_particle(self, particle_idx):
        """Remove a particle"""
        buffer = self.buffers[0]
        if not (0 <= particle_idx < buffer["particle_count"]):
            return False
        if not buffer["particles"]["active"][particle_idx]:
            return False

        # Clear spatial grid
        x = buffer["particles"]["x"][particle_idx]
        y = buffer["particles"]["y"][particle_idx]

        if (
            0 <= x < self.grid_width
            and 0 <= y < self.grid_height
            and buffer["spatial_grid"][x, y] == particle_idx
        ):
            buffer["spatial_grid"][x, y] = -1

        # Mark as inactive
        buffer["particles"]["active"][particle_idx] = False

        if not self._dirty_texture:
            self.mark_particles_dirty()

        return True

    def move_particle(self, particle_idx, new_x, new_y):
        """Move a particle to a new position"""
        if not (0 <= particle_idx < self.buffers[0]["particle_count"]):
            return False
        if not self.buffers[0]["particles"]["active"][particle_idx]:
            return False
        if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height):
            return False
        if self.buffers[0]["spatial_grid"][new_x, new_y] != -1:
            return False  # Position occupied

        # Clear old position
        old_x, old_y = (
            self.buffers[0]["particles"]["x"][particle_idx],
            self.buffers[0]["particles"]["y"][particle_idx],
        )
        if 0 <= old_x < self.grid_width and 0 <= old_y < self.grid_height:
            self.buffers[0]["spatial_grid"][old_x, old_y] = -1

        # Set new position
        self.buffers[0]["particles"]["x"][particle_idx] = new_x
        self.buffers[0]["particles"]["y"][particle_idx] = new_y
        self.buffers[0]["spatial_grid"][new_x, new_y] = particle_idx

        if not self._dirty_texture:
            self.mark_particles_dirty()

        return True

    def place_particles(self, pos, shape, width, height, element_name):
        """Place multiple particles in a shape"""
        grid_x = (pos[0] - self.x) / self.pixel_size
        grid_y = int((pos[1] - self.y) / self.pixel_size)
        half_w = int(width / self.pixel_size / 2)
        half_h = int(height / self.pixel_size / 2)

        particles_added = 0
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                x, y = grid_x + dx, grid_y + dy
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    if shape == "square" or self._is_in_shape(
                        x, y, grid_x, grid_y, shape, half_w, half_h
                    ):
                        if self.add_particle(x, y, element_name):
                            particles_added += 1

        return particles_added

    def _is_in_shape(
        self,
        x: int,
        y: int,
        center_x: int,
        center_y: int,
        shape: Tuple,
        half_w: float,
        half_h: float,
    ):
        """Check if point is inside the given shape"""
        if shape == "ellipse":
            if half_w <= 0 or half_h <= 0:
                return x == center_x and y == center_y
            dx = (x - center_x) / half_w
            dy = (y - center_y) / half_h
            return dx * dx + dy * dy <= 1
        elif shape == "triangle":
            y_rel = y - center_y
            x_rel = abs(x - center_x)
            return (
                y_rel >= -half_h
                and y_rel <= half_h
                and x_rel <= (half_h - y_rel) * half_w / half_h
                if half_h > 0
                else False
            )

        return True

    def update(self, dt: float):
        """Modified update to mark dirty when physics runs"""
        # print(
        #     f"Update called: particle_count={self.buffers[0]['particle_count']}, dirty={self._dirty_texture}"
        # )
        # print(
        #     f"Update called: particle_count={self.buffers[0]['particle_count']}, dirty={self._dirty_texture}"
        # )
        # print(f"  Widget: pos=({self.x},{self.y}) size=({self.width},{self.height})")

        if self.buffers[0]["particle_count"] > 0:
            self.mark_particles_dirty()

        stats = update_gravity(self, self.buffers[0], dt)
        self.render()

        if hasattr(self, "just_switched_edge_mode") and self.just_switched_edge_mode:
            self.just_switched_edge_mode = False

        # In your update method
        # if random.randint(0, 180) == 0:  # Every ~3  F821
        #     self.validate_spatial_grid()

    def debug_spatial_grid(self, limit=10):
        buffer = self.buffers[0]
        p = buffer["particles"]
        s = buffer["spatial_grid"]
        n = buffer["particle_count"]
        print("---- Grid Snapshot ----")
        for i in range(min(limit, n)):
            if not p["active"][i]:
                continue
            x, y = int(p["x"][i]), int(p["y"][i])
            vy = p["velocity_y"][i]
            below = s[x, y - 1] if 0 <= y - 1 < s.shape[1] else "OOB"
            print(f"i={i:3} (x={x:3}, y={y:3}) vy={vy:6.2f} below={below}")
            if y == 0 and vy != 0 and self._edge_mode == 0:
                print(
                    f"WARNING: Particle {i} at y=0 with non-zero vy={vy} in Normal mode"
                )

    def _setup_texture_rendering(self, width, height):
        """Initialize texture rendering with dirty tracking"""
        from kivy.graphics.texture import Texture

        # Only recreate texture if dimensions changed
        if (
            self.particle_texture is None
            or self.particle_texture.width != width
            or self.particle_texture.height != height
        ):
            self.particle_texture = Texture.create(size=(width, height))
            self.particle_texture.mag_filter = "nearest"
            self.particle_texture.min_filter = "nearest"

            # Pre-allocate texture data - keep the same array
            self.texture_data = np.zeros((height, width, 4), dtype=np.uint8)

        self.mesh = self._create_mesh()
        self._dirty_texture = True

    def _reset_graph(self):
        """Remove explicit gc.collect() call"""
        max_particles = self.grid_width * self.grid_height
        self.buffers[0] = self._create_buffer(max_particles)
        self.particle_texture = None
        self.mesh = None
        self._setup_texture_rendering(self.grid_width, self.grid_height)
        self.canvas.clear()

        if self.buffers[0]["particle_count"] == 0:
            return True

    def get_particle_at(self, x, y):
        """Get particle index at position - supports both (x, y) and pos tuple"""
        grid_x = int((x - self.x) / self.pixel_size)
        grid_y = int((y - self.y) / self.pixel_size)

        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return -1

        particle_idx = self.buffers[0]["spatial_grid"][grid_x, grid_y]
        if particle_idx == -1:
            return particle_idx
        return self._get_particle_data(particle_idx)

    def _get_particle_data(self, particle_idx):
        """Get all data for a particle by index"""
        if (
            not (0 <= particle_idx < self.buffers[0]["particle_count"])
            or not self.buffers[0]["particles"]["active"][particle_idx]
        ):
            return None

        return self.buffers[0]["particles"][particle_idx]

    def compact_arrays(self):
        """Remove gaps in particle arrays (call periodically for optimization)"""
        buffer = self.buffers[0]
        if buffer["particle_count"] == 0:
            return

        # Find active particles
        active_mask = buffer["particles"]["active"][: buffer["particle_count"]]
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == buffer["particle_count"]:
            return  # No gaps to compact

        # Clear spatial grid
        buffer["spatial_grid"].fill(-1)

        # Compact arrays (copy active to front)
        new_count = len(active_indices)
        buffer["particles"][:new_count] = buffer["particles"][active_indices]

        # Clear inactive entries
        buffer["particles"]["active"][new_count:] = False

        # Rebuild spatial grid
        for i in range(new_count):
            x = int(buffer["particles"]["x"][i])
            y = int(buffer["particles"]["y"][i])
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                buffer["spatial_grid"][x, y] = i

        buffer["particle_count"] = new_count

        Logger.debug(f"Compacted arrays: {new_count} active particles remain")

    def rebuild_spatial_grid(self):
        """Fully rebuild spatial grid from particle positions and active mask."""
        buffer = self.buffers[0]
        particles = buffer["particles"]
        spatial_grid = buffer["spatial_grid"]
        particle_count = buffer["particle_count"]

        # Clear grid
        spatial_grid.fill(-1)

        # Refill based on current particle positions
        for i in range(particle_count):
            if not particles["active"][i]:
                continue
            x = int(particles["x"][i])
            y = int(particles["y"][i])
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                spatial_grid[x, y] = i

    def _update_canvas(self):
        """Separate canvas update to avoid clearing unnecessarily"""
        # print("_update_canvas called")
        # print(f"  mesh exists: {self.mesh is not None}")
        # print(f"  texture exists: {self.particle_texture is not None}")
        # print(f"  widget pos: ({self.x}, {self.y})")
        # print(f"  widget size: ({self.width}, {self.height})")

        if not self.mesh or not self.particle_texture:
            print("  Skipping - missing mesh or texture")
            return

        # Only clear and redraw if texture actually changed
        if self._dirty_texture:
            self.canvas.clear()

            with self.canvas:
                PushMatrix()
                from kivy.graphics import Scale, Translate

                Translate(self.x, self.y)
                Scale(self.pixel_size, self.pixel_size, 1)

                w, h = self.grid_width, self.grid_height
                vertices = [0, 0, 0, 0, w, 0, 1, 0, w, h, 1, 1, 0, h, 0, 1]
                self.mesh.vertices = vertices
                self.mesh.texture = self.particle_texture
                self.canvas.add(self.mesh)
                PopMatrix()

    def debug_grid_state(self):
        """Print debug information about the grid state"""
        active_count = np.sum(self.active[: self.particle_count])
        spatial_particles = np.sum(self.spatial_grid != -1)

        Logger.debug("=== GRID DEBUG ===")
        Logger.debug(f"Grid size: {self.grid_width}x{self.grid_height}")
        Logger.debug(f"Total particle slots: {len(self.active)}")
        Logger.debug(f"Particle count: {self.particle_count}")
        Logger.debug(f"Active particles: {active_count}")
        Logger.debug(f"Spatial grid particles: {spatial_particles}")

        if active_count != spatial_particles:
            Logger.warning("WARNING: Active count != spatial grid count!")

        # Check for spatial grid consistency
        mismatches = 0
        for i in range(self.particle_count):
            if self.active[i]:
                x, y = self.x_coords[i], self.y_coords[i]
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    if self.spatial_grid[x, y] != i:
                        mismatches += 1

        if mismatches > 0:
            Logger.warning(f"WARNING: {mismatches} spatial grid mismatches!")

        Logger.debug("================")

    def _create_mesh(self):
        """Create a single mesh that covers the entire simulation area"""
        # Use grid dimensions, not widget size
        w, h = self.grid_width, self.grid_height

        vertices = [
            0,
            0,
            0,
            0,  # bottom-left
            w,
            0,
            1,
            0,  # bottom-right
            w,
            h,
            1,
            1,  # top-right
            0,
            h,
            0,
            1,  # top-left
        ]

        indices = [0, 1, 2, 2, 3, 0]

        mesh = Mesh(vertices=vertices, indices=indices, mode="triangles")
        mesh.texture = self.particle_texture
        return mesh

    def mark_particles_dirty(self):
        """Call this when particles move/change to mark texture as needing update"""
        self._dirty_texture = True

    # def render(self):
    #     if not self._dirty_texture:
    #         print("Render skipped - not dirty")
    #         return
    #
    #     print(f"Rendering: particle_count={self.buffers[0]['particle_count']}")
    #
    #     # Simple debug render
    #     self.texture_data.fill(0)
    #
    #     grid = self.buffers[0]["spatial_grid"]
    #     particles = self.buffers[0]["particles"]
    #
    #     occupied_cells = 0
    #     # For each cell with a particle
    #     for x in range(self.grid_width):
    #         for y in range(self.grid_height):
    #             idx = grid[x, y]
    #             if idx >= 0 and idx < self.buffers[0]["particle_count"]:
    #                 if particles["active"][idx]:
    #                     occupied_cells += 1
    #                     # White pixel
    #                     self.texture_data[y, x] = [255, 255, 255, 255]
    #
    #     print(f"Occupied cells: {occupied_cells}")
    #     print(f"Texture shape: {self.texture_data.shape}")
    #     print(f"Grid dimensions: {self.grid_width}x{self.grid_height}")
    #
    #     self.particle_texture.blit_buffer(
    #         self.texture_data.tobytes(),
    #         colorfmt='rgba',
    #         bufferfmt='ubyte'
    #     )
    #     self.particle_texture.flip_vertical()
    #     self._update_canvas()
    #     self._dirty_texture = False
    #     print("Render complete")
    #
    def render(self):
        """Optimized render method with dirty checking and array reuse"""
        if (
            self.particle_texture is None
            or self.buffers[0]["particle_count"] == 0
            or not self._dirty_texture
        ):
            return

        particle_count = self.buffers[0]["particle_count"]

        # Ensure render arrays are allocated
        self._allocate_render_arrays(particle_count)

        # Clear texture only once (reuse the same array)
        self.texture_data.fill(0)

        # Use pre-allocated arrays to avoid repeated allocations
        particles = self.buffers[0]["particles"]
        active_slice = particles["active"][:particle_count]
        active_count = np.sum(active_slice)

        if active_count == 0:
            self._dirty_texture = False
            return

        # Reuse pre-allocated arrays instead of creating new ones
        active_indices_full = np.where(active_slice)[0]
        active_count = len(active_indices_full)
        if active_count == 0:
            self._dirty_texture = False
            return
        active_indices = active_indices_full

        # Extract positions and colors in batch (reuse arrays)
        positions = self._temp_positions[:active_count]
        colors = self._temp_colors[:active_count]

        positions[:, 0] = particles["x"][active_indices]
        positions[:, 1] = particles["y"][active_indices]

        # Vectorized color conversion (avoid astype which creates new array)
        temp_colors = particles["color"][active_indices]
        np.multiply(temp_colors, 255, out=colors, casting="unsafe")

        # Bounds checking (vectorized)
        valid_mask = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < self.grid_width)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < self.grid_height)
        )

        if not np.any(valid_mask):
            self._dirty_texture = False
            return

        valid_positions = positions[valid_mask]
        valid_colors = colors[valid_mask]

        if self.texture_data is not None:
            # Batch texture update (much faster than individual assignments)
            self.texture_data[valid_positions[:, 1], valid_positions[:, 0]] = (
                valid_colors
            )

        # Upload to GPU only when needed
        try:
            self.particle_texture.blit_buffer(
                self.texture_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
            )
            self._update_canvas()
            self._dirty_texture = False  # Mark as clean
        except Exception as e:
            Logger.error(f"Texture upload failed: {e}")

    def get_stats(self):
        """Get simulation statistics"""
        active_count = np.sum(self.active[: self.particle_count])
        burning_count = np.sum(
            self.burning[: self.particle_count] & self.active[: self.particle_count]
        )
        avg_temp = (
            np.mean(
                self.temperatures[: self.particle_count][
                    self.active[: self.particle_count]
                ]
            )
            if active_count > 0
            else 0
        )

        return {
            "active_particles": active_count,
            "total_created": self.particle_count,
            "burning_particles": burning_count,
            "average_temperature": avg_temp,
        }

    def validate_spatial_grid(self):
        """Check if spatial grid matches particle positions"""
        particles = self.buffers[0]["particles"]
        spatial_grid = self.buffers[0]["spatial_grid"]
        particle_count = self.buffers[0]["particle_count"]

        errors = 0
        for i in range(particle_count):
            if not particles["active"][i]:
                continue

            x = int(particles["x"][i])
            y = int(particles["y"][i])

            # Check if particle is where it thinks it is
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                occupant = spatial_grid[x, y]
                if occupant != i:
                    errors += 1
                    Logger.error(
                        f"DESYNC: Particle {i} at ({x},{y}) but grid says {occupant}"
                    )

        # Check reverse - grid points to valid particles
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                idx = spatial_grid[x, y]
                if idx != -1 and idx < particle_count:
                    if particles["active"][idx]:
                        px = int(particles["x"][idx])
                        py = int(particles["y"][idx])
                        if px != x or py != y:
                            errors += 1
                            Logger.error(
                                f"DESYNC: Grid[{x},{y}] points to particle {idx} but it's at ({px},{py})"
                            )

        if errors > 0:
            Logger.critical(f"Found {errors} spatial grid desyncs!")
        return errors
