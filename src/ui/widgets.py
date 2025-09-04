# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalContextManager=false
# pyright: reportOptionalMemberAccess=false
from engine.components.gravity import update_gravity
import numpy as np

# import pymunk
from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import PopMatrix, PushMatrix, Mesh
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

# from scipy.signal import convolve2d


class BackgroundColor:
    background_color = ListProperty([1, 1, 1, 1])


class PickerLabel(Label):
    background_color = ColorProperty([0.9, 0.9, 0.9, 1])
    index = NumericProperty(0)

    def __init__(self, **kwargs):
        super(PickerLabel, self).__init__(**kwargs)
        self.bind(index=self._update_index)

    def _update_index(self, instance, value):
        column = self.parent
        if column:
            column.remove_widget(self)
            column.add_widget(self, index=value)

    def on_color_touch(self, touch):
        if self.collide_point(*touch.pos) and not self.text:
            Logger.info(f"Clicked color: {self.background_color}")
            popup = Popup(
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


# class Particle(Widget):
#     id = StringProperty("")
#     symbol = StringProperty("")
#     mass = NumericProperty(0.0)
#     density = NumericProperty(0.0)
#     state = StringProperty("")
#     color = ColorProperty([1.0, 1.0, 1.0, 1.0])
#     radius = NumericProperty(1.0)
#     specific_heat = NumericProperty(1.0)
#     heat_conductivity = NumericProperty(0.0)
#     electrical_conductivity = NumericProperty(0.0)
#     elasticity = NumericProperty(0.0)
#     friction = NumericProperty(0.0)
#     ignition_temperature = NumericProperty(0.0)
#     flame_temperature = NumericProperty(0.0)
#     burn_duration = NumericProperty(0.0)
#     oxygen_requirement = NumericProperty(0.0)
#     phase_transitions = DictProperty({})
#     combustion_products = ListProperty([])
#     temperature = NumericProperty(0.0)
#     pressure = NumericProperty(0.0)
#     velocity = ListProperty([0.0, 0.0])
#     acceleration = ListProperty([0.0, 0.0])
#     energy = NumericProperty(0.0)
#     current = NumericProperty(0.0)
#     burning = BooleanProperty(False)
#     burn_progress = NumericProperty(0.0)
#     dynamic_color = ColorProperty([0.0, 0.0, 0.0, 0.0])
#     x = NumericProperty(0)
#     y = NumericProperty(0)
#
#     def __init__(self, material: str, x: int = 0, y: int = 0, **kwargs):
#         super(Particle, self).__init__(**kwargs)
#         __slots__ = [
#             "id",
#             "symbol",
#             "mass",
#             "density",
#             "state",
#             "color",
#             "radius",
#             "specific_heat",
#             "heat_conductivity",
#             "electrical_conductivity",
#             "elasticity",
#             "friction",
#             "ignition_temperature",
#             "flame_temperature",
#             "burn_duration",
#             "oxygen_requirement",
#             "phase_transitions",
#             "combustion_products",
#             "temperature",
#             "pressure",
#             "velocity",
#             "acceleration",
#             "energy",
#             "current",
#             "burning",
#             "burn_progress",
#             "dynamic_color",
#             "x",
#             "y",
#             "reactivity",
#             "propagation",
#             "_app",
#         ]
#         self._app = App.get_running_app()
#
#         # Fast property assignment
#         props = self._app.elements[material]
#         intrinsic = props["intrinsic_properties"]
#         dynamic = props["dynamic_properties"]
#         interaction = props["interaction_properties"]
#
#         # Direct assignment for speed
#         self.id = intrinsic["id"]
#         self.symbol = intrinsic["symbol"]
#         self.mass = intrinsic["mass"]
#         self.density = intrinsic["density"]
#         self.state = intrinsic["state"]
#         self.color = intrinsic["color"]
#         self.radius = intrinsic["radius"]
#         self.specific_heat = intrinsic["specific_heat"]
#         self.heat_conductivity = intrinsic["heat_conductivity"]
#         self.electrical_conductivity = intrinsic["electrical_conductivity"]
#         self.elasticity = intrinsic["elasticity"]
#         self.friction = intrinsic["friction"]
#         self.ignition_temperature = intrinsic.get("ignition_temperature", 0.0)
#         self.flame_temperature = intrinsic.get("flame_temperature", 0.0)
#         self.burn_duration = intrinsic.get("burn_duration", 0.0)
#         self.oxygen_requirement = intrinsic.get("oxygen_requirement", 0.0)
#         self.phase_transitions = intrinsic.get("phase_transitions", {})
#         self.combustion_products = intrinsic.get("combustion_products", [])
#         self.temperature = dynamic["temperature"]
#         self.pressure = dynamic["pressure"]
#         self.velocity = dynamic["velocity"]
#         self.acceleration = dynamic["acceleration"]
#         self.energy = dynamic["energy"]
#         self.current = dynamic["current"]
#         self.burning = dynamic["burning"]
#         self.burn_progress = dynamic["burn_progress"]
#         self.dynamic_color = dynamic.get("color", [0.0, 0.0, 0.0, 0.0])
#         self.reactivity = interaction["reactivity"]
#         self.propagation = interaction["propagation"]
#         self.x = x
#         self.y = y


STATES = {"solid": 0, "liquid": 1, "gas": 2}


class SimulationGrid(Widget):
    def __init__(self, pixel_size=4, **kwargs):
        super().__init__(**kwargs)

        self.pixel_size = pixel_size
        self.on_resized = None
        self.is_initialized = False
        self.is_ready = False

        # Element definitions (map element names to IDs)
        self.element_map: dict = {}
        self.element_properties = {}
        self._init_elements()
        Clock.schedule_once(self._check_initialization, 0.1)

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
            ("state", np.int32),
            ("velocity_x", np.float32),
            ("velocity_y", np.float32),
            ("active", bool),
            ("burning", bool),
            ("melting", bool),
            ("color", np.float32, 4),
        ]
        return {
            "particles": np.zeros(max_particles, dtype=dtype),
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
            self.buffers[0]["particles"]["mass"][idx] = intrinsic.get("mass", 1.0)
            self.buffers[0]["particles"]["temperature"][idx] = dynamic.get(
                "temperature", 20.0
            )
            self.buffers[0]["particles"]["state"][idx] = STATES[
                intrinsic.get("state", 0)
            ]  # 0=solid,1=liquid,2=gas

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

        # Update spatial grid
        self.buffers[0]["spatial_grid"][x, y] = idx
        self.buffers[0]["particle_count"] += 1

        return True

    def remove_particle(self, particle_idx):
        """Remove a particle"""
        if not (0 <= particle_idx < self.particle_count):
            return False
        if not self.active[particle_idx]:
            return False

        # Clear spatial grid
        x, y = self.x_coords[particle_idx], self.y_coords[particle_idx]
        if (
            0 <= x < self.grid_width
            and 0 <= y < self.grid_height
            and self.spatial_grid[x, y] == particle_idx
        ):
            self.spatial_grid[x, y] = -1

        # Mark as inactive
        self.active[particle_idx] = False

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

        return True

    def place_particles(self, pos, shape, width, height, element_name):
        """Place multiple particles in a shape"""
        grid_x = int((pos[0] - self.x) / self.pixel_size)
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

    def _is_in_shape(self, x, y, center_x, center_y, shape, half_w, half_h):
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

    # def update(self, dt):
    #     """
    #     Fixed update method that properly modifies particle positions
    #     """
    #     time_step = dt
    #
    #     # Initialize physics system only once (add this to __init__ instead)
    #     if not hasattr(self, "gravity_physics"):
    #         from engine.components.gravity import OptimizedGravityPhysics
    #
    #         self.gravity_physics = OptimizedGravityPhysics(
    #             self.grid_width, self.grid_height, self.buffers[0]["particles"].size
    #         )
    #
    #     # Get particle count and check if we have any particles
    #     particle_count = self.buffers[0]["particle_count"]
    #     if particle_count == 0:
    #         self.render()
    #         return
    #
    #     # Get active particles
    #     active_mask = self.buffers[0]["particles"]["active"][:particle_count]
    #     active_count = np.sum(active_mask)
    #
    #     if active_count == 0:
    #         self.render()
    #         return
    #
    #     print(f"Updating {active_count} active particles")
    #
    #     # === GRAVITY PHYSICS ===
    #     # Step 1: Apply gravity to velocities
    #     from engine.components.gravity import apply_gravity_vectorized
    #
    #     apply_gravity_vectorized(
    #         self.buffers[0]["particles"]["velocity_y"][:particle_count],
    #         active_mask,
    #         time_step,
    #     )
    #
    #     # Step 2: Process falling with collision detection
    #     from engine.components.gravity import batch_gravity_fall_fixed
    #
    #     fall_count = batch_gravity_fall_fixed(
    #         self.buffers[0]["particles"]["x"][:particle_count],
    #         self.buffers[0]["particles"]["y"][:particle_count],
    #         self.buffers[0]["particles"]["velocity_y"][:particle_count],
    #         active_mask,
    #         self.buffers[0]["spatial_grid"],
    #         self.grid_height,
    #         time_step,
    #     )
    #
    #     # Step 3: Handle diagonal sliding for blocked particles
    #     from engine.components.gravity import diagonal_slide_fixed
    #
    #     slide_count = diagonal_slide_fixed(
    #         self.buffers[0]["particles"]["x"][:particle_count],
    #         self.buffers[0]["particles"]["y"][:particle_count],
    #         self.buffers[0]["particles"]["velocity_x"][:particle_count],
    #         self.buffers[0]["particles"]["velocity_y"][:particle_count],
    #         active_mask,
    #         self.buffers[0]["spatial_grid"],
    #         self.grid_width,
    #         self.grid_height,
    #         self.buffers[0]["particles"]["state"][:particle_count],
    #         time_step,
    #     )
    #
    #     if fall_count > 0 or slide_count > 0:
    #         print(f"Physics: {fall_count} particles fell, {slide_count} particles slid")
    #
    #     # === THERMAL UPDATES (simplified) ===
    #     hot_mask = (
    #         self.buffers[0]["particles"]["temperature"][:particle_count] > 25.0
    #     ) & active_mask
    #     if np.any(hot_mask):
    #         self.buffers[0]["particles"]["temperature"][:particle_count][hot_mask] -= (
    #             0.1 * time_step
    #         )
    #
    #     # === COMBUSTION ===
    #     combustible = (
    #         self.buffers[0]["particles"]["temperature"][:particle_count] > 100.0
    #     ) & active_mask
    #     if np.any(combustible):
    #         self.buffers[0]["particles"]["burning"][:particle_count][combustible] = True
    #         self.buffers[0]["particles"]["temperature"][:particle_count][
    #             combustible
    #         ] = 800.0
    #         burning_indices = np.where(combustible)[0]
    #         self.buffers[0]["particles"]["color"][burning_indices] = [
    #             1.0,
    #             0.5,
    #             0.0,
    #             1.0,
    #         ]
    #
    #     self.render()
    #
    def update(self, dt):
        stats = update_gravity(self, self.buffers[0])
        self.render()

        # Periodic compaction
        # if not hasattr(self, "frame_counter"):
        #     self.frame_counter = 0
        # self.frame_counter += 1
        # if self.frame_counter % 60 == 0:
        #     self.compact_arrays()

    # Apply gravity to velocities ( g=pixel scale)
    # g = 4.0
    # self.buffers[0]["particles"]["velocity_y"][active_indices] -= g * time_step
    #
    # # Filter solid particles
    # solid_mask = (
    #     self.buffers[0]["particles"]["state"][: self.buffers[0]["particle_count"]][
    #         active_mask
    #     ]
    #     == 0
    # )
    # solid_indices = active_indices[solid_mask]

    # === PHYSICS UPDATES ===

    # Vectorized straight fall (as before, but modulated by velocity)
    # can_fall = self._can_particles_fall(solid_indices)
    # falling_particles = solid_indices[can_fall]
    #
    # if len(falling_particles) > 0:
    #     old_x = self.buffers[0]["particles"]["x"][falling_particles]
    #     old_y = self.buffers[0]["particles"]["y"][falling_particles].copy()
    #     new_y = np.round(
    #         old_y + self.buffers[0]["particles"]["velocity_y"][falling_particles]
    #     ).astype(np.int16)
    #
    #     # Clip to bounds
    #     new_y = np.maximum(new_y, 0)
    #     valid_fall = new_y < old_y
    #     falling_particles = falling_particles[valid_fall]
    #     old_x, old_y, new_y = (
    #         old_x[valid_fall],
    #         old_y[valid_fall],
    #         new_y[valid_fall],
    #     )
    #
    #     if len(falling_particles) > 0:
    #         self.buffers[0]["spatial_grid"][old_x, old_y] = -1
    #         self.buffers[0]["spatial_grid"][old_x, new_y] = falling_particles
    #         self.buffers[0]["particles"]["y"][falling_particles] = new_y
    #
    # # For blocked solids: Diagonal slide for natural piling (loop over sorted for no conflicts)
    # blocked_indices = solid_indices[~can_fall]
    # if len(blocked_indices) > 0:
    #     # Sort by y descending (top-first, y high to low) to allow upper particles to slide after lower
    #     sort_order = np.argsort(-self.buffers[0]["particles"]["y"][blocked_indices])
    #     blocked_indices = blocked_indices[sort_order]
    #
    #     for idx in blocked_indices:
    #         x, y = (
    #             self.buffers[0]["particles"]["x"][idx],
    #             self.buffers[0]["particles"]["y"][idx],
    #         )
    #         if y <= 0:
    #             continue
    #         below_y = y - 1
    #
    #         # Randomize direction preference for natural look
    #         dirs = [-1, 1] if random.random() < 0.5 else [1, -1]
    #
    #         moved = False
    #         for dx in dirs:
    #             new_x = int(x) + dx  # Cast to int to avoid uint16 issues
    #             # Check bounds *before* moving
    #             if (
    #                 0 <= new_x < self.grid_width
    #                 and below_y >= 0
    #                 and self.buffers[0]["spatial_grid"][new_x, below_y] == -1
    #             ):
    #                 self.move_particle(idx, new_x, below_y)
    #                 moved = True
    #                 break
    #
    #         if moved:
    #             # Dampen velocity on slide
    #             self.buffers[0]["particles"]["velocity_y"][idx] *= 0.8
    #
    # # === THERMAL UPDATES ===
    # hot_mask = (
    #     self.buffers[0]["particles"]["temperature"][
    #         : self.buffers[0]["particle_count"]
    #     ]
    #     > 25.0
    # )
    # hot_particles = active_mask & hot_mask
    # if np.any(hot_particles):
    #     self.buffers[0]["particles"]["temperature"][
    #         : self.buffers[0]["particle_count"]
    #     ][hot_particles] -= 0.1 * dt
    #
    # # === COMBUSTION ===
    # combustible = (
    #     self.buffers[0]["particles"]["temperature"][
    #         : self.buffers[0]["particle_count"]
    #     ]
    #     > 100.0
    # ) & active_mask
    # if np.any(combustible):
    #     self.buffers[0]["burning"][: self.buffers[0]["particle_count"]][
    #         combustible
    #     ] = True
    #     self.self.buffers[0]["particles"]["temperature"][
    #         : self.buffers[0]["particle_count"]
    #     ][combustible] = 800.0
    #     burning_indices = np.where(combustible)[0]
    #     self.self.buffers[0]["particles"]["color"][burning_indices] = [
    #         1.0,
    #         0.5,
    #         0.0,
    #         1.0,
    #     ]  # Orange

    # Integrate batch processing
    # self._batch_process_burning_particles()
    # self._batch_process_conduction()

    # def _spawn_combustion_products(self, x, y, particle):
    #     """Spawn combustion products"""
    #     elements = self.app.elements
    #
    #     for product in particle.combustion_products:
    #         if product["id"] in elements:
    #             new_particle = Particle(product["id"], x=x, y=y)
    #             self.grid[x, y] = new_particle
    #             self._categorize_particle(new_particle, x, y)
    #             return  # Only spawn first product for now

    def _categorize_particle(self, idx, x, y):
        """Categorize particles for optimized batch processing"""
        element_id = self.element_ids[idx]
        if element_id in self.element_properties:
            props = self.element_properties[element_id][1]
            # Simplified checks (adjust based on your needs)
            if self.burning[idx] or props.get("ignition_temperature", 0) > 0:
                if (x, y) not in self._burning_particles:
                    self._burning_particles.append((x, y))
            if props.get("heat_conductivity", 0) > 0:
                if (x, y) not in self._conductive_particles:
                    self._conductive_particles.append((x, y))
            if props.get("reactivity", {}).get("compatibility"):
                if (x, y) not in self._reactive_particles:
                    self._reactive_particles.append((x, y))

    def _batch_process_burning_particles(self):
        """Process all burning particles in batch"""
        if not self._burning_particles:
            return

        to_remove = []
        for x, y in list(self._burning_particles):
            idx = self.spatial_grid[x, y]
            if idx == -1 or not self.active[idx]:
                to_remove.append((x, y))
                continue

            # Simplified ignition check
            if self._should_ignite(idx, x, y):
                self.burning[idx] = True
                self.temperatures[idx] = self.element_properties[self.element_ids[idx]][
                    1
                ].get("flame_temperature", 800.0)
                self.colors[idx] = [1.0, 0.8, 0.4, 0.8]

            if self.burning[idx]:
                # Simplified burn-out logic (e.g., remove after burning)
                if self.temperatures[idx] > 1000.0:  # Arbitrary threshold
                    self.remove_particle(idx)
                    to_remove.append((x, y))

        for pos in to_remove:
            self._burning_particles.remove(pos)

    def _should_ignite(self, idx, x, y):
        """Check if particle should ignite"""
        props = self.element_properties[self.element_ids[idx]][1]
        return (
            props.get("ignition_temperature", 0) > 0
            and self.temperatures[idx] >= props.get("ignition_temperature", 0)
            and not self.burning[idx]
            # Add air_grid oxygen check when implemented
        )

    def _propagate_heat_fast(self, x, y, particle):
        """Fast heat propagation using numpy operations"""
        if not hasattr(particle, "propagation"):
            return

        heat_amount = particle.flame_temperature * 0.1

        # Get neighbor positions
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in self._neighbor_offsets
            if 0 <= x + dx < self.grid.shape[0] and 0 <= y + dy < self.grid.shape[1]
        ]

        for nx, ny in neighbors:
            neighbor = self.grid[nx, ny]
            if neighbor and hasattr(neighbor, "temperature"):
                heat_transfer = min(
                    heat_amount * 0.25,
                    (particle.temperature - neighbor.temperature) * 0.1,
                )
                if heat_transfer > 0:
                    neighbor.temperature += heat_transfer / neighbor.specific_heat

    def _batch_process_conduction(self):
        """Vectorized heat and electrical conduction"""
        if not self._conductive_particles:
            return

        # Process in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(self._conductive_particles), chunk_size):
            chunk = self._conductive_particles[i : i + chunk_size]
            self._process_conduction_chunk(chunk)

    def _process_conduction_chunk(self, particle_positions):
        """Process conduction for a chunk of particles"""
        for x, y in particle_positions:
            particle = self.grid[x, y]
            if not particle:
                continue

            # Get valid neighbors
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in self._neighbor_offsets
                if (
                    0 <= x + dx < self.grid.shape[0]
                    and 0 <= y + dy < self.grid.shape[1]
                    and self.grid[x + dx, y + dy] is not None
                )
            ]

            if not neighbors:
                continue

            # Batch process heat transfer
            for nx, ny in neighbors:
                neighbor = self.grid[nx, ny]
                if not neighbor:
                    continue

                # Heat conduction
                temp_diff = particle.temperature - neighbor.temperature
                if abs(temp_diff) > 0.1:  # Skip tiny transfers
                    heat_transfer = (
                        particle.heat_conductivity
                        * particle.propagation.get("heat_conduction_rate", 0.1)
                        * temp_diff
                        * 0.1
                    )
                    if abs(heat_transfer) > 0.01:
                        neighbor.temperature += heat_transfer / neighbor.specific_heat
                        particle.temperature -= heat_transfer / particle.specific_heat

    def _batch_process_interactions(self):
        """Process particle interactions using spatial partitioning"""
        if not self._reactive_particles:
            return

        # Clear interaction cells
        self._interaction_cells.clear()

        # Partition reactive particles
        for x, y in self._reactive_particles:
            cell_x = x // self._interaction_grid_size
            cell_y = y // self._interaction_grid_size
            self._interaction_cells[(cell_x, cell_y)].append((x, y))

        # Process interactions within each cell and adjacent cells
        for (cell_x, cell_y), particles in self._interaction_cells.items():
            # Check adjacent cells too
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    adj_cell = (cell_x + dx, cell_y + dy)
                    if adj_cell in self._interaction_cells:
                        self._process_cell_interactions(
                            particles, self._interaction_cells[adj_cell]
                        )

    def _process_cell_interactions(self, particles1, particles2):
        """Process interactions between particles in two cells"""
        for x1, y1 in particles1:
            particle1 = self.grid[x1, y1]
            if not particle1:
                continue

            for x2, y2 in particles2:
                if x1 == x2 and y1 == y2:  # Same particle
                    continue

                # Only check adjacent particles
                if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
                    continue

                particle2 = self.grid[x2, y2]
                if particle2:
                    self._evaluate_interaction_fast(
                        x1, y1, particle1, x2, y2, particle2
                    )

    def _evaluate_interaction_fast(self, x1, y1, p1, x2, y2, p2):
        """Fast interaction evaluation with early exits"""
        if not hasattr(p1, "reactivity") or p2.id not in p1.reactivity.get(
            "compatibility", {}
        ):
            return

        reaction = p1.reactivity["compatibility"][p2.id]
        thresholds = p1.reactivity.get("thresholds", {})

        # Quick temperature/pressure check
        if (
            p1.temperature < thresholds.get("temperature", [0])[0]
            or p1.pressure < thresholds.get("pressure", [0])[0]
        ):
            return

        # Probability check
        if np.random.random() >= reaction.get("reaction_probability", 0):
            return

        # Apply reaction effects (simplified)
        self._apply_reaction_effects(p1, p2, reaction)

    def _apply_reaction_effects(self, p1, p2, reaction):
        """Apply reaction effects efficiently"""
        # Apply deltas to particles
        for key, value in reaction.get("deltas", {}).get("self", {}).items():
            if hasattr(p1, key):
                if isinstance(value, str) and value.startswith("+"):
                    setattr(p1, key, getattr(p1, key) + float(value[1:]))
                else:
                    setattr(p1, key, value)

    def apply_force(self, x, y, reaction):
        force = reaction["force_magnitude"]
        radius = reaction["blast_radius"]
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                    distance = ((dx**2 + dy**2) ** 0.5) + 0.1
                    if distance <= radius:
                        impulse = force / distance
                        if self.grid[nx, ny]:
                            self.grid[nx, ny].velocity[0] += impulse * dx
                            self.grid[nx, ny].velocity[1] += impulse * dy
                        if (nx, ny) in self.rigid_bodies:
                            body = self.rigid_bodies[(nx, ny)]
                            body.apply_impulse_at_local_point(
                                (impulse * dx, impulse * dy), (0, 0)
                            )

    def _can_particles_fall(self, particle_indices):
        if particle_indices.size == 0:
            return np.array([], dtype=bool)

        # Get coordinates for active particles
        xs = self.buffers[0]["particles"]["x"][particle_indices]
        ys = self.buffers[0]["particles"]["y"][particle_indices]
        below_ys = ys - 1

        # Full bounds check: ensure x, y are within grid and below_y is valid
        valid_coords = (
            (xs >= 0)
            & (xs < self.grid_width)
            & (ys > 0)
            & (ys < self.grid_height)
            & (below_ys >= 0)
        )

        # Initialize can_fall array
        can_fall = np.zeros(len(particle_indices), dtype=bool)

        # Process only valid particles
        valid_indices = np.where(valid_coords)[0]
        if len(valid_indices) > 0:
            # Check if space below is empty
            below_empty = (
                self.buffers[0]["spatial_grid"][
                    xs[valid_indices], below_ys[valid_indices]
                ]
                == -1
            )
            can_fall[valid_indices] = below_empty

        return can_fall

    def propagate_conductivity_vectorized(self):
        for x, y in (
            self.buffers[0]["particles"]["x"],
            self.buffers[0]["particles"]["y"],
        ):
            particle = self.buffers[0]["spatial_grid"][x, y]
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if 0 <= x + dx < self.grid.shape[0] and 0 <= y + dy < self.grid.shape[1]
            ]
            for nx, ny in neighbors:
                if self.buffers[0]["spatial_grid"][nx, ny]:
                    neighbor = self.buffers[0]["spatial_grid"][nx, ny]
                    heat_transfer = (
                        particle.heat_conductivity
                        * particle.propagation["heat_conduction_rate"]
                        * (particle.temperature - neighbor.temperature)
                        * 0.1
                    )
                    neighbor.temperature += heat_transfer / neighbor.specific_heat
                    particle.temperature -= heat_transfer / particle.specific_heat

    def _setup_texture_rendering(self, width, height):
        """Initialize texture rendering"""
        from kivy.graphics.texture import Texture

        self.particle_texture = Texture.create(size=(width, height))
        self.particle_texture.mag_filter = "nearest"
        self.particle_texture.min_filter = "nearest"

        # Pre-allocate texture data
        self.texture_data = np.zeros((height, width, 4), dtype=np.uint8)
        self.mesh = self._create_mesh()

    def render(self):
        if self.particle_texture is None or self.buffers[0]["particle_count"] == 0:
            return

        # Clear texture
        self.texture_data.fill(0)

        # Get active particles
        active_mask = self.buffers[0]["particles"]["active"][
            : self.buffers[0]["particle_count"]
        ]
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return

        # Vectorized assignment
        xs = self.buffers[0]["particles"]["x"][active_indices]
        ys = self.buffers[0]["particles"]["y"][active_indices]
        colors = (self.buffers[0]["particles"]["color"][active_indices] * 255).astype(
            np.uint8
        )  # Shape: (N, 4)

        # Bounds check (optional, but prevents index errors)
        valid = (xs >= 0) & (xs < self.grid_width) & (ys >= 0) & (ys < self.grid_height)
        xs, ys, colors = xs[valid], ys[valid], colors[valid]

        xs_int = xs.astype(np.int32)
        ys_int = ys.astype(np.int32)

        self.texture_data[ys_int, xs_int] = colors

        # Upload to GPU
        try:
            self.particle_texture.blit_buffer(
                self.texture_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
            )
        except Exception as e:
            Logger.error(f"Texture upload failed: {e}")
            return

        self.canvas.clear()

        with self.canvas:
            PushMatrix()
            from kivy.graphics import Scale, Translate

            Translate(self.x, self.y)
            Scale(self.pixel_size, self.pixel_size, 1)

            if self.mesh:
                w, h = self.grid_width, self.grid_height
                vertices = [0, 0, 0, 0, w, 0, 1, 0, w, h, 1, 1, 0, h, 0, 1]
                self.mesh.vertices = vertices
                self.mesh.texture = self.particle_texture
                self.canvas.add(self.mesh)
            PopMatrix()

    def _reset_graph(self):
        max_particles = self.grid_width * self.grid_height
        self.buffers[0] = self._create_buffer(max_particles)
        self.canvas.clear()
        if self.buffers[0]["particle_count"] == 0:
            return True

    def resize(self, new_width, new_height):
        """Resize the simulation grid and preserve existing particles"""
        old_width, old_height = self.grid_width, self.grid_height

        if new_width == old_width and new_height == old_height:
            return

        Logger.debug(
            f"Resizing grid from {old_width}x{old_height} to {new_width}x{new_height}"
        )

        # Store old particle data
        old_particle_count = self.particle_count
        old_active = self.active[:old_particle_count].copy()
        old_x = self.x_coords[:old_particle_count].copy()
        old_y = self.y_coords[:old_particle_count].copy()
        old_element_ids = self.element_ids[:old_particle_count].copy()
        old_temperatures = self.temperatures[:old_particle_count].copy()
        old_masses = self.masses[:old_particle_count].copy()
        old_colors = self.colors[:old_particle_count].copy()
        old_states = self.states[:old_particle_count].copy()
        old_burning = self.burning[:old_particle_count].copy()
        old_melting = self.melting[:old_particle_count].copy()
        old_velocities_x = self.velocities_x[:old_particle_count].copy()
        old_velocities_y = self.velocities_y[:old_particle_count].copy()

        # Update grid dimensions
        self.grid_width = new_width
        self.grid_height = new_height

        # Resize all arrays
        max_particles = new_width * new_height

        # Recreate arrays with new size
        self.element_ids = np.zeros(max_particles, dtype=np.uint16)
        self.x_coords = np.zeros(max_particles, dtype=np.uint16)
        self.y_coords = np.zeros(max_particles, dtype=np.uint16)
        self.temperatures = np.full(max_particles, 20.0, dtype=np.float32)
        self.states = np.full(max_particles, -1, dtype=np.int32)
        self.masses = np.ones(max_particles, dtype=np.float32)
        self.densities = np.ones(max_particles, dtype=np.float32)
        self.velocities_x = np.zeros(max_particles, dtype=np.float32)
        self.velocities_y = np.zeros(max_particles, dtype=np.float32)
        self.active = np.zeros(max_particles, dtype=bool)
        self.burning = np.zeros(max_particles, dtype=bool)
        self.melting = np.zeros(max_particles, dtype=bool)
        self.colors = np.ones((max_particles, 4), dtype=np.float32)

        # Recreate spatial grid
        self.spatial_grid = np.full((new_width, new_height), -1, dtype=np.int32)

        # Transfer particles that fit in the new grid
        new_particle_count = 0
        for i in range(old_particle_count):
            if not old_active[i]:
                continue
            x, y = old_x[i], old_y[i]
            if (
                0 <= x < new_width
                and 0 <= y < new_height
                and self.spatial_grid[x, y] == -1
            ):
                idx = new_particle_count
                self.element_ids[idx] = old_element_ids[i]
                self.x_coords[idx] = x
                self.y_coords[idx] = y
                self.temperatures[idx] = old_temperatures[i]
                self.masses[idx] = old_masses[i]
                self.colors[idx] = old_colors[i]
                self.states[idx] = old_states[i]
                self.burning[idx] = old_burning[i]
                self.melting[idx] = old_melting[i]
                self.velocities_x[idx] = old_velocities_x[i]
                self.velocities_y[idx] = old_velocities_y[i]
                self.active[idx] = True
                new_particle_count += 1

        self.particle_count = new_particle_count

        # Recreate air grid (if you still use it)
        if hasattr(self, "_air_overlay"):
            self._air_overlay.resize(new_width, new_height)

        # Recreate rendering components
        self._setup_texture_rendering(new_width, new_height)

        Logger.debug(f"Resize complete: {new_particle_count} particles transferred")

        # Call resize callback if set
        if self.on_resized:
            self.on_resized()

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
        if self.particle_count == 0:
            return

        # Find active particles
        active_mask = self.active[: self.particle_count]
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == self.particle_count:
            return  # No gaps to compact

        # Clear spatial grid
        self.spatial_grid.fill(-1)

        # Compact arrays
        new_count = len(active_indices)

        self.element_ids[:new_count] = self.element_ids[active_indices]
        self.x_coords[:new_count] = self.x_coords[active_indices]
        self.y_coords[:new_count] = self.y_coords[active_indices]
        self.temperatures[:new_count] = self.temperatures[active_indices]
        self.masses[:new_count] = self.masses[active_indices]
        self.colors[:new_count] = self.colors[active_indices]
        self.burning[:new_count] = self.burning[active_indices]
        self.melting[:new_count] = self.melting[active_indices]
        self.velocities_x[:new_count] = self.velocities_x[active_indices]
        self.velocities_y[:new_count] = self.velocities_y[active_indices]
        self.active[:new_count] = True

        # Clear inactive entries
        self.active[new_count:] = False

        # Rebuild spatial grid
        for i in range(new_count):
            x, y = self.x_coords[i], self.y_coords[i]
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                self.spatial_grid[x, y] = i

        self.particle_count = new_count

        Logger.debug(f"Compacted arrays: {len(active_indices)} active particles remain")

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
