# pyright: reportAttributeAccessIssue=false
import logging
import math

import numpy as np
import pymunk
from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import Mesh, PopMatrix, PushMatrix
from kivy.graphics.texture import Texture
from kivy.properties import (
    BooleanProperty,
    ColorProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.behaviors import TouchRippleBehavior
from kivy.uix.label import Label
from kivy.uix.label import Label as PopupLabel
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from scipy.signal import convolve2d

from app.ui.elements import load_elements

ELEMENTS = load_elements()


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
            logging.info(f"Clicked color: {self.background_color}")
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
            logging.debug(f"Touch down at: {touch.pos}")
            self.ripple_show(touch)
            return True
        return super(RippleButton, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            logging.debug(f"Touch up at: {touch.pos}")
            self.ripple_fade()
            app = App.get_running_app()
            if app:
                app.open_modal()
            return True
        return super(RippleButton, self).on_touch_up(touch)


class Particle(Widget):
    id = StringProperty("")
    symbol = StringProperty("")
    mass = NumericProperty(0.0)
    density = NumericProperty(0.0)
    state = StringProperty("")
    color = ColorProperty([1.0, 1.0, 1.0, 1.0])
    radius = NumericProperty(1.0)
    specific_heat = NumericProperty(1.0)
    heat_conductivity = NumericProperty(0.0)
    electrical_conductivity = NumericProperty(0.0)
    elasticity = NumericProperty(0.0)
    friction = NumericProperty(0.0)
    ignition_temperature = NumericProperty(0.0)
    flame_temperature = NumericProperty(0.0)
    burn_duration = NumericProperty(0.0)
    oxygen_requirement = NumericProperty(0.0)
    phase_transitions = DictProperty({})
    combustion_products = ListProperty([])
    temperature = NumericProperty(0.0)
    pressure = NumericProperty(0.0)
    velocity = ListProperty([0.0, 0.0])
    acceleration = ListProperty([0.0, 0.0])
    energy = NumericProperty(0.0)
    current = NumericProperty(0.0)
    burning = BooleanProperty(False)
    burn_progress = NumericProperty(0.0)
    dynamic_color = ColorProperty([0.0, 0.0, 0.0, 0.0])
    x = NumericProperty(0)
    y = NumericProperty(0)

    def __init__(self, material: str, x: int = 0, y: int = 0, **kwargs):
        super(Particle, self).__init__(**kwargs)
        intrinsic = ELEMENTS[material]["intrinsic_properties"]
        dynamic = ELEMENTS[material]["dynamic_properties"]
        interaction = ELEMENTS[material]["interaction_properties"]
        self.id = intrinsic["id"]
        self.symbol = intrinsic["symbol"]
        self.mass = intrinsic["mass"]
        self.density = intrinsic["density"]
        self.state = intrinsic["state"]
        self.color = intrinsic["color"]
        self.radius = intrinsic["radius"]
        self.specific_heat = intrinsic["specific_heat"]
        self.heat_conductivity = intrinsic["heat_conductivity"]
        self.electrical_conductivity = intrinsic["electrical_conductivity"]
        self.elasticity = intrinsic["elasticity"]
        self.friction = intrinsic["friction"]
        self.ignition_temperature = intrinsic.get("ignition_temperature", 0.0)
        self.flame_temperature = intrinsic.get("flame_temperature", 0.0)
        self.burn_duration = intrinsic.get("burn_duration", 0.0)
        self.oxygen_requirement = intrinsic.get("oxygen_requirement", 0.0)
        self.phase_transitions = intrinsic.get("phase_transitions", {})
        self.combustion_products = intrinsic.get("combustion_products", [])
        self.temperature = dynamic["temperature"]
        self.pressure = dynamic["pressure"]
        self.velocity = dynamic["velocity"]
        self.acceleration = dynamic["acceleration"]
        self.energy = dynamic["energy"]
        self.current = dynamic["current"]
        self.burning = dynamic["burning"]
        self.burn_progress = dynamic["burn_progress"]
        self.dynamic_color = dynamic.get("color", [0.0, 0.0, 0.0, 0.0])
        self.reactivity = interaction["reactivity"]
        self.propagation = interaction["propagation"]
        self.x = x
        self.y = y


class SimulationGrid(Widget):
    cursor = ObjectProperty(None)

    def __init__(self, width=100, height=100, pixel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.particles = []
        self.mesh = None
        self.particle_texture = None
        self.texture_data = None
        Clock.schedule_once(self._bind_scene_graph, 0)
        self.space = pymunk.Space()
        self.grid = np.zeros((width, height), dtype=object)
        self.grid[:] = None
        self.air_grid = np.zeros(
            (width, height),
            dtype=[("oxygen", float), ("temp", float), ("pressure", float)],
        )
        self.air_grid["oxygen"] = 0.21
        self.air_grid["temp"] = 20.0
        self.air_grid["pressure"] = 1.0
        self.elements = ELEMENTS
        self.rigid_bodies = {}
        self.pixel_size = pixel_size
        self.on_resized = None
        self.active_particles = set()
        self._setup_texture_rendering(width, height)

    def _setup_texture_rendering(self, width, height):
        """Initialize the texture and mesh for particle rendering"""
        self.grid_width = width
        self.grid_height = height

        # Create texture for particle data (RGBA format)
        self.particle_texture = Texture.create(size=(width, height))
        self.particle_texture.mag_filter = "nearest"  # Pixel-perfect rendering
        self.particle_texture.min_filter = "nearest"

        # Create numpy array for texture data (height, width, 4) for RGBA
        self.texture_data = np.zeros((height, width, 4), dtype=np.uint8)

        # Create mesh covering the entire simulation area
        # This will be updated when the widget is resized
        self._create_mesh()

    def _create_mesh(self):
        """Create a single mesh that covers the entire simulation area"""
        if not hasattr(self, "size") or not self.size[0] or not self.size[1]:
            # Widget not fully initialized yet
            return

        w, h = self.size

        # Create vertices for a single quad covering the entire area
        # Format: [x, y, u, v] where u,v are texture coordinates
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

        # Indices for two triangles making a quad
        indices = [0, 1, 2, 2, 3, 0]

        self.mesh = Mesh(vertices=vertices, indices=indices, mode="triangles")
        self.mesh.texture = self.particle_texture

    def _bind_scene_graph(self, dt):
        app = App.get_running_app()
        if app:
            scene_graph = app.root.ids.scene_graph
            self.bind_to_scene_graph(scene_graph)
        else:
            logging.error(
                "App instance not found, we cannot continue on like this. I'm sorry. It's not you, it's me."
            )
            exit(1)

    def bind_to_scene_graph(self, scene_graph):
        def update_grid(*args):
            self.size = scene_graph.size
            self.pos = scene_graph.pos
            width = math.ceil(scene_graph.width / self.pixel_size)
            height = math.ceil(scene_graph.height / self.pixel_size)
            self.resize(width, height)

        scene_graph.bind(size=update_grid, pos=update_grid)

    def add_particle(self, x, y, element, color=None):
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
            particle = Particle(element, x=x, y=y)
            body = pymunk.Body()
            body.position = (x * self.pixel_size, y * self.pixel_size)
            shape = pymunk.Circle(body, particle.radius * self.pixel_size)
            shape.mass = particle.mass
            shape.elasticity = particle.elasticity
            shape.friction = particle.friction
            if color:
                particle.color = color
            self.space.add(body, shape)
            self.rigid_bodies[(x, y)] = body
            # self.particles.append(particle)
            self.grid[x, y] = particle
            self.active_particles.add((x, y))
        else:
            print(f"Failed to add particle at ({x}, {y}): Out of bounds")

    def get_particle_at(self, pos):
        grid_x = int((pos[0] - self.x) / self.pixel_size)
        grid_y = int((pos[1] - self.y) / self.pixel_size)
        if 0 <= grid_x < self.grid.shape[0] and 0 <= grid_y < self.grid.shape[1]:
            return self.grid[grid_x, grid_y]
        return None

    def place_particles(self, pos, shape, width, height, element_id):
        if element_id not in ELEMENTS:
            print(f"Invalid element_id: {element_id}")
            return
        grid_x = int((pos[0] - self.x) / self.pixel_size)
        grid_y = int((pos[1] - self.y) / self.pixel_size)
        half_w = int(width / self.pixel_size / 2)
        half_h = int(height / self.pixel_size / 2)
        added = 0
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                x, y = grid_x + dx, grid_y + dy
                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    if shape == "square" or self.is_in_shape(
                        x, y, grid_x, grid_y, shape, half_w, half_h
                    ):
                        self.add_particle(x, y, element_id)
                        added += 1
        logging.info(
            f"Placed {added} particles at ({grid_x}, {grid_y}), shape={shape}, element={element_id}"
        )
        self.render()  # Immediate render for testing

    def is_in_shape(self, x, y, center_x, center_y, shape, half_w, half_h):
        if shape == "ellipse":
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
            )
        return True

    def resize(self, new_width, new_height):
        old_width, old_height = self.grid.shape
        if new_width != old_width or new_height != old_height:
            self.grid = np.full((new_width, new_height), None)
            self.air_grid = np.zeros(
                (new_width, new_height),
                dtype=[("oxygen", float), ("temp", float), ("pressure", float)],
            )
            self.air_grid["oxygen"] = 0.21
            self.air_grid["temp"] = 20.0
            self.air_grid["pressure"] = 1.0

            # Resize texture data
            self._setup_texture_rendering(new_width, new_height)
            new_particles = []
            new_active = set()

            for p in self.particles:
                if 0 <= p.x < new_width and 0 <= p.y < new_height:
                    self.grid[p.x, p.y] = p
                    new_particles.append(p)
                    new_active.add((p.x, p.y))

            self.particles = new_particles
            self.active_particles = new_active

            # Recreate mesh with new dimensions
            self._create_mesh()

            if self.on_resized:
                self.on_resized()

    def update(self, dt):
        self.space.step(dt)
        new_grid = np.copy(self.grid)
        self.update_air(dt)

        for x, y in list(self.active_particles):
            if self.grid[x, y]:
                self.process_particle(x, y, new_grid)
            else:
                self.active_particles.discard((x, y))

        self.propagate_conductivity_vectorized()

        self.grid = new_grid
        self.render()

    def process_particle(self, x, y, new_grid):
        particle = self.grid[x, y]
        self.check_phase_transition(particle, x, y)
        self.check_combustion(x, y, particle, new_grid)
        # self.propagate_conductivity(x, y, particle, new_grid)
        neighbors = [
            (x + 1, y) if x + 1 < self.grid.shape[0] else None,
            (x - 1, y) if x - 1 >= 0 else None,
            (x, y + 1) if y + 1 < self.grid.shape[1] else None,
            (x, y - 1) if y - 1 >= 0 else None,
        ]
        for neighbor in neighbors:
            if neighbor is not None:
                nx, ny = neighbor
                if nx is not None and ny is not None and self.grid[nx, ny]:
                    self.evaluate_interaction(
                        x, y, particle, nx, ny, self.grid[nx, ny], new_grid
                    )

    def update_pressure(self, x, y, particle):
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if 0 <= x + dx < self.grid.shape[0] and 0 <= y + dy < self.grid.shape[1]
        ]
        neighbor_count = sum(1 for nx, ny in neighbors if self.grid[nx, ny])
        particle.pressure = (
            1.0 + 0.1 * neighbor_count * particle.density
        )  # Simplified model

    def update_velocity(self, particle, dt):
        body = self.rigid_bodies.get((particle.x, particle.y))
        if body:
            body.apply_force_at_local_point(
                (0, -9.81 * particle.mass), (0, 0)
            )  # Gravity
            particle.velocity = list(body.velocity / self.pixel_size)

    def check_phase_transition(self, particle, x, y):
        transitions = particle.phase_transitions
        current_temp = particle.temperature
        current_pressure = particle.pressure

        # Linear interpolation for phase transition thresholds
        for transition_type, points in transitions.items():
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                if p1["pressure"] <= current_pressure <= p2["pressure"]:
                    # Interpolate temperature threshold
                    t1, t2 = p1["temperature"], p2["temperature"]
                    p1_p, p2_p = p1["pressure"], p2["pressure"]
                    threshold_temp = t1 + (current_pressure - p1_p) * (t2 - t1) / (
                        p2_p - p1_p
                    )
                    if current_temp <= threshold_temp:
                        particle.state = (
                            "liquid" if transition_type == "condensation" else "solid"
                        )
                        # Apply phase change effects
                        effects = particle.propagation["phase_change_effects"][
                            transition_type
                        ]
                        particle.pressure += float(effects["pressure"].lstrip("+"))
                        particle.temperature += float(effects["temperature"])
                        particle.current = float(effects["current"])
                        # Adjust physics properties
                        if particle.state == "liquid":
                            particle.friction *= 2
                            particle.velocity = [v * 0.5 for v in particle.velocity]
                        elif particle.state == "solid":
                            particle.friction *= 10
                            particle.velocity = [0, 0]
                        return
        particle.state = "gas"  # Default or revert to gas

    def check_combustion(self, x, y, particle, new_grid):
        if not particle.ignition_temperature:
            return
        if (
            particle.temperature >= particle.ignition_temperature
            and self.air_grid[x, y]["oxygen"] >= particle.oxygen_requirement
            and not particle.burning
        ):
            particle.burning = True
            particle.temperature = particle.flame_temperature
            particle.dynamic_color = [1.0, 0.8, 0.4, 0.8]
        if particle.burning:
            particle.burn_progress = min(
                particle.burn_progress + 0.1 / particle.burn_duration, 1.0
            )
            particle.mass -= 0.1 * particle.mass / particle.burn_duration
            self.air_grid[x, y]["oxygen"] -= 0.05 * particle.oxygen_requirement
            self.air_grid[x, y]["temp"] += 0.1 * particle.flame_temperature
            if particle.burn_progress >= 1.0:
                self.spawn_products(x, y, particle.combustion_products, new_grid)
                new_grid[x, y] = None
            else:
                self.propagate_combustion_effects(x, y, particle, new_grid)

    def propagate_combustion_effects(self, x, y, particle, new_grid):
        bias = particle.propagation["heat_direction_bias"]
        neighbors = [
            ("up", x, y + 1, bias["upward"]),
            ("down", x, y - 1, bias["downward"]),
            ("left", x - 1, y, bias["horizontal"]),
            ("right", x + 1, y, bias["horizontal"]),
        ]
        for direction, nx, ny, weight in neighbors:
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                temp_gradient = (
                    self.air_grid[nx, ny]["temp"] - self.air_grid[x, y]["temp"]
                )
                press_gradient = (
                    self.air_grid[nx, ny]["pressure"] - self.air_grid[x, y]["pressure"]
                )
                weight *= 1.0 + 0.1 * temp_gradient - 0.05 * press_gradient
                if self.grid[nx, ny]:
                    neighbor = self.grid[nx, ny]
                    if (
                        neighbor is not None
                        and hasattr(neighbor, "temperature")
                        and hasattr(neighbor, "specific_heat")
                    ):
                        heat_transfer = (
                            particle.heat_conductivity
                            * weight
                            * (particle.flame_temperature - neighbor.temperature)
                            * 0.1
                        )
                        neighbor.temperature += heat_transfer / neighbor.specific_heat
                        self.air_grid[nx, ny]["temp"] += heat_transfer * 0.05
                        self.air_grid[nx, ny]["pressure"] += 0.5 * weight

    def update_air(self, dt):
        # Copy the air grid to avoid modifying in place
        new_air = np.copy(self.air_grid)

        # Extract oxygen field
        oxygen = self.air_grid["oxygen"]

        # Define the Laplacian kernel for diffusion (3x3)
        kernel = (
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) * 0.01
        )  # Scale by diffusion rate

        # Compute diffusion using convolution
        net_change = convolve2d(
            oxygen, kernel, mode="same", boundary="fill", fillvalue=0
        )

        # Update oxygen values
        new_air["oxygen"] += net_change

        # Ensure oxygen values stay non-negative
        new_air["oxygen"] = np.clip(new_air["oxygen"], 0, np.inf)

        # Assign back to self.air_grid
        self.air_grid = new_air

    def spawn_products(self, x, y, products, new_grid):
        for product in products:
            if product["id"] in self.elements:
                new_grid[x, y] = Particle(product["id"], x=x, y=y)

    def evaluate_interaction(self, x, y, particle, nx, ny, neighbor, new_grid):
        if neighbor.id in particle.reactivity["compatibility"]:
            reaction = particle.reactivity["compatibility"][neighbor.id]
            if (
                particle.temperature
                >= particle.reactivity["thresholds"]["temperature"][0]
                and particle.pressure
                >= particle.reactivity["thresholds"]["pressure"][0]
            ):
                if np.random.random() < reaction["reaction_probability"]:
                    for key, value in reaction["deltas"]["self"].items():
                        if hasattr(particle, key):
                            if isinstance(value, str) and value.startswith("+"):
                                setattr(
                                    particle,
                                    key,
                                    getattr(particle, key) + float(value[1:]),
                                )
                            else:
                                setattr(particle, key, value)
                    for key, value in reaction["deltas"]["neighbor"].items():
                        if hasattr(neighbor, key):
                            if isinstance(value, str) and value.startswith("+"):
                                setattr(
                                    neighbor,
                                    key,
                                    getattr(neighbor, key) + float(value[1:]),
                                )
                            else:
                                setattr(neighbor, key, value)
                    self.apply_force(x, y, reaction)
                    self.spawn_products(x, y, reaction["products"], new_grid)

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

    def propagate_conductivity_vectorized(self):
        for x, y in self.active_particles:
            particle = self.grid[x, y]
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if 0 <= x + dx < self.grid.shape[0] and 0 <= y + dy < self.grid.shape[1]
            ]
            for nx, ny in neighbors:
                if self.grid[nx, ny]:
                    neighbor = self.grid[nx, ny]
                    heat_transfer = (
                        particle.heat_conductivity
                        * particle.propagation["heat_conduction_rate"]
                        * (particle.temperature - neighbor.temperature)
                        * 0.1
                    )
                    neighbor.temperature += heat_transfer / neighbor.specific_heat
                    particle.temperature -= heat_transfer / particle.specific_heat

    def propagate_conductivity(self, x, y, particle, new_grid):
        neighbors = [
            (x + 1, y) if x + 1 < self.grid.shape[0] else None,
            (x - 1, y) if x - 1 >= 0 else None,
            (x, y + 1) if y + 1 < self.grid.shape[1] else None,
            (x, y - 1) if y - 1 >= 0 else None,
        ]
        for neighbor in neighbors:
            if neighbor is not None:
                nx, ny = neighbor
                if nx is not None and ny is not None and self.grid[nx, ny]:
                    neighbor_particle = self.grid[nx, ny]
                    heat_transfer = (
                        particle.heat_conductivity
                        * particle.propagation["heat_conduction_rate"]
                        * (particle.temperature - neighbor_particle.temperature)
                        * 0.1
                    )
                    neighbor_particle.temperature += (
                        heat_transfer / neighbor_particle.specific_heat
                    )
                    particle.temperature -= heat_transfer / particle.specific_heat
                    curr_transfer = (
                        particle.electrical_conductivity
                        * particle.propagation["electrical_conduction_rate"]
                        * (particle.current - neighbor_particle.current)
                        * 0.01
                    )
                    neighbor_particle.current += curr_transfer
                    particle.current -= curr_transfer

    def render(self):
        if (
            not self.canvas
            or self.particle_texture is None
            or self.texture_data is None
        ):
            logging.error("Canvas or texture not initialized")
            return

        # Clear texture data (set all pixels to transparent)
        self.texture_data.fill(0)

        # Update texture with particle data
        for x, y in self.active_particles:
            particle = self.grid[x, y]
            if particle is None:
                logging.warning(f"None particle at ({x}, {y})")
                self.active_particles.discard((x, y))
                continue

            # Get particle color (dynamic color takes precedence)
            color = (
                particle.dynamic_color
                if hasattr(particle, "dynamic_color") and particle.dynamic_color[3] > 0
                else particle.color
            )

            if len(color) != 4 or color[3] <= 0:
                logging.warning(f"Invalid/transparent color at ({x}, {y}): {color}")
                color = [1, 0, 0, 1]  # Fallback to opaque red

            # Convert float color [0-1] to uint8 [0-255] and set pixel
            # Note: texture coordinates are (y, x) due to how images are stored
            if (
                0 <= y < self.texture_data.shape[0]
                and 0 <= x < self.texture_data.shape[1]
            ):
                self.texture_data[y, x] = [
                    int(color[0] * 255),  # R
                    int(color[1] * 255),  # G
                    int(color[2] * 255),  # B
                    int(color[3] * 255),  # A
                ]

        # Update the texture with new data
        try:
            self.particle_texture.blit_buffer(
                self.texture_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
            )
        except Exception as e:
            logging.error(f"Texture update failed: {e}")
            return

        # Clear canvas and render the textured mesh
        self.canvas.clear()

        with self.canvas:
            PushMatrix()
            # Transform to widget position
            from kivy.graphics import Scale, Translate

            Translate(self.x, self.y)

            # Scale the texture to match pixel size
            Scale(self.pixel_size, self.pixel_size, 1)

            if self.mesh:
                # Update mesh vertices to match current size
                w = self.grid_width
                h = self.grid_height

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

                self.mesh.vertices = vertices
                self.mesh.texture = self.particle_texture
                self.canvas.add(self.mesh)

            PopMatrix()

        # logging.debug(f"Texture rendered: {len(self.active_particles)} particles")
