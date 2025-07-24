# pyright: reportAttributeAccessIssue=false
import copy
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import pymunk
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import Color, Rectangle
from kivy.properties import (
    BooleanProperty,
    ColorProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.behaviors.touchripple import TouchRippleBehavior
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.widget import Widget


def load_elements(elements_dir="data/elements"):
    elements = {}
    elements_path = Path(elements_dir)
    if not elements_path.exists():
        logging.error(f"Elements data directory not found: {elements_dir}")
        return elements

    for file_path in elements_path.glob("*.json"):
        try:
            with file_path.open("r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"File {file_path.name} does not contain a dictionary"
                    )
                for key in data.keys():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Non-string key found in {file_path.name}: {key}"
                        )
                logging.info(f"Loaded from {file_path.name}: {list(data.keys())}")
                elements.update(data)
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error in {file_path.name}: {e}")
    logging.info(f"ELEMENTS keys: {list(elements.keys())}")
    return elements


ELEMENTS = load_elements()


class BackgroundColor:
    background_color = ListProperty([1, 1, 1, 1])


class PickerLabel(Label):
    background_color = ListProperty([0.5, 0.5, 0.5, 1])


class TableHead(Label, BackgroundColor):
    pass


class PickerModal(ModalView):
    _cols = ObjectProperty({})

    def __init__(self, **kwargs):
        super(PickerModal, self).__init__(**kwargs)
        self._cols = {}

    def on_pre_open(self):
        # Clear existing widgets
        self.ids.picker_table.clear_widgets()
        self._cols = {}

        # Collect all unique properties to display as columns
        properties = set()
        for _, element in ELEMENTS.items():
            for k in element["intrinsic_properties"].keys():
                if k in [
                    "radius",
                    "phase_transitions",
                    "ignition_temperature",
                    "flame_temperature",
                    "burn_duration",
                    "oxygen_requirement",
                    "combustion_products",
                ]:
                    continue
                properties.add(k)
        properties = sorted(properties)  # Sort for consistent column order

        # Set number of columns in GridLayout
        self.ids.picker_table.cols = len(properties)

        # Add header row
        for prop in properties:
            label = TableHead(text=prop)
            self._cols[prop] = [label]
            self.ids.picker_table.add_widget(label)

        # Add data rows
        for _, element in sorted(ELEMENTS.items()):
            for prop in properties:
                if prop == "color":
                    # Create PickerLabel for color column with background color
                    color_value = element["intrinsic_properties"].get(
                        prop, [1, 1, 1, 1]
                    )
                    # Validate color_value
                    if not isinstance(color_value, list) or len(color_value) != 4:
                        logging.warning(
                            f"Invalid color for element {element['intrinsic_properties']['id']}: {color_value}"
                        )
                        color_value = [0.74, 0.72, 0.42, 1.0]  # Default to dark khaki
                    label = Factory.PickerLabel(text="", background_color=color_value)
                    self._cols[prop].append(label)
                    self.ids.picker_table.add_widget(label)
                else:
                    # Create PickerLabel for other columns
                    value = element["intrinsic_properties"].get(prop, "")
                    label = Factory.PickerLabel(text=str(value))
                    self._cols[prop].append(label)
                    self.ids.picker_table.add_widget(label)

        logging.debug(f"Table columns: {properties}")


class RippleButton(TouchRippleBehavior, Label):
    """A button that creates a ripple effect when pressed."""

    def __init__(self, **kwargs):
        super(RippleButton, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        """Handle touch down event to create a ripple effect."""
        if self.collide_point(*touch.pos):
            logging.debug(f"Touch down at: {touch.pos}")
            self.ripple_show(touch)
            return True
        return super(RippleButton, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        """Handle touch up event to stop the ripple effect."""
        if self.collide_point(*touch.pos):
            logging.debug(f"Touch up at: {touch.pos}")
            self.ripple_fade()
            app = App.get_running_app()
            if app:
                app.open_modal()
            return True
        return super(RippleButton, self).on_touch_up(touch)


class Cursor:
    pos = ObjectProperty([0, 0])

    def __init__(self):
        Window.bind(on_mouse_pos=self.update_pos)
        self.selection = None

    def update_pos(self, instance, pos):
        self.pos = pos


class Particle(Widget):
    # Common intrinsic properties
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
    ignition_temperature = NumericProperty(0.0)  # Optional, 0.0 for for non-combustible
    flame_temperature = NumericProperty(0.0)
    burn_duration = NumericProperty(0.0)
    oxygen_requirement = NumericProperty(0.0)
    phase_transitions = DictProperty({})
    combustion_products = ListProperty([])

    # Common dynamic properties
    temperature = NumericProperty(0.0)
    pressure = NumericProperty(0.0)
    velocity = ListProperty([0.0, 0.0])
    acceleration = ListProperty([0.0, 0.0])
    energy = NumericProperty(0.0)
    current = NumericProperty(0.0)
    burning = BooleanProperty(False)
    burn_progress = NumericProperty(0.0)
    dynamic_color = ColorProperty([0.0, 0.0, 0.0, 0.0])  # Overrides color when burning

    # Interaction properties
    reactivity = DictProperty({})
    propagation = DictProperty({})

    def __init__(self, material: str, **kwargs):
        super(Particle, self).__init__(**kwargs)
        # Initialize from JSON definition
        intrinsic = ELEMENTS[material]["intrinsic_properties"]
        dynamic = ELEMENTS[material]["dynamic_properties"]
        interaction = ELEMENTS[material]["interaction_properties"]

        # Intrinsic properties
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

        # Dynamic properties
        self.temperature = dynamic["temperature"]
        self.pressure = dynamic["pressure"]
        self.velocity = dynamic["velocity"]
        self.acceleration = dynamic["acceleration"]
        self.energy = dynamic["energy"]
        self.current = dynamic["current"]
        self.burning = dynamic["burning"]
        self.burn_progress = dynamic["burn_progress"]
        self.dynamic_color = dynamic.get("color", [0.0, 0.0, 0.0, 0.0])

        # Interaction properties
        self.reactivity = interaction["reactivity"]
        self.propagation = interaction["propagation"]


class SimulationGrid(Widget):
    def __init__(self, width=100, height=100, pixel_size=1, **kwargs):
        super().__init__(**kwargs)

        self.particles = []

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

        self.elements = ELEMENTS  # Your JSON definitions

        self.rigid_bodies = {}

        self.pixel_size = pixel_size
        self.on_resize = None

    def _bind_scene_graph(self, dt):
        scene_graph = self.parent.ids.scene_graph
        self.bind_to_scene_graph(scene_graph)

    def bind_to_scene_graph(self, scene_graph):
        def update_grid(*args):
            self.size = scene_graph.size
            self.pos = scene_graph.pos

            width = math.ceil(scene_graph.width / self.pixel_size)
            height = math.ceil(scene_graph.height / self.pixel_size)
            self.resize(width, height)

        scene_graph.bind(size=update_grid, pos=update_grid)
        update_grid()

    def add_particle(self, x, y, element, color=None):
        particle = Particle(element)
        self.particles.append(particle)
        self.grid[x, y] = particle

    def resize(self, new_width, new_height):
        old_width, old_height = self.grid.shape
        if new_width != old_width or new_height != old_height:
            self.grid = np.full((new_width, new_height), None)
            self.air_grid = np.zeros(
                (new_width, new_height),
                dtype=[("oxygen", float), ("temp", float), ("pressure", float)],
            )
            # Re-place all particles, removing those out of bounds
            new_particles = []
            for p in self.particles:
                if 0 <= p.x < new_width and 0 <= p.y < new_height:
                    self.grid[p.x, p.y] = p
                    new_particles.append(p)
            self.particles = new_particles
            if self.on_resized:
                self.on_resized()

    def update(self, dt):
        self.space.step(dt)
        new_grid = np.copy(self.grid)
        self.update_air(dt)
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y]:
                    self.process_particle(x, y, new_grid)
        self.grid = new_grid
        self.render()

    def process_particle(self, x, y, new_grid):
        particle = self.grid[x, y]
        self.check_combustion(x, y, particle, new_grid)
        self.propagate_conductivity(x, y, particle, new_grid)
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
            particle.dynamic_color = [1.0, 0.8, 0.4, 0.8]  # Flame color
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
        new_air = np.copy(self.air_grid)
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for nx, ny in neighbors:
                    if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                        oxygen_diff = (
                            self.air_grid[nx, ny]["oxygen"]
                            - self.air_grid[x, y]["oxygen"]
                        ) * 0.01
                        new_air[x, y]["oxygen"] += oxygen_diff
                        new_air[nx, ny]["oxygen"] -= oxygen_diff
        self.air_grid = new_air

    def spawn_products(self, x, y, products, new_grid):
        for product in products:
            if product["id"] in self.elements:
                new_grid[x, y] = Particle(copy.deepcopy(self.elements[product["id"]]))

    def evaluate_interaction(self, x, y, particle, nx, ny, neighbor, new_grid):
        if neighbor.id in particle.reactivity["compatibility"]:
            reaction = particle.reactivity["compatibility"][neighbor.id]
            if (
                particle.temperature >= reaction["thresholds"]["temperature"][0]
                and particle.pressure >= reaction["thresholds"]["pressure"][0]
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
        if not self.canvas:
            return
        self.canvas.clear()
        pixel_size = self.pixel_size
        x_offset, y_offset = self.pos  # Use grid offset
        with self.canvas:
            for x, y in np.ndindex(self.grid.shape):
                particle = self.grid[x, y]
                if particle is not None:
                    color = (
                        particle.dynamic_color
                        if particle.dynamic_color[3] > 0
                        else particle.color
                    )
                    Color(*color)
                    Rectangle(
                        pos=(x_offset + x * pixel_size, y_offset + y * pixel_size),
                        size=(pixel_size, pixel_size),
                    )


class ParticleSimApp(App):
    def build(self):
        return self.root

    def on_start(self):
        colors = [
            [
                random.random(),
                random.random(),
                random.random(),
                1,
            ]
            for _ in range(200)
        ]
        if self.root and self.root.ids.simulation_grid:
            simulation_grid = self.root.ids.simulation_grid
            element_keys = list(ELEMENTS.keys())
            if not element_keys:
                raise ValueError("No elements loaded into ELEMENTS dictionary")

            def add_test_particles():
                for color in colors:
                    x = random.randint(0, simulation_grid.grid.shape[0] - 2)
                    y = random.randint(0, simulation_grid.grid.shape[1] - 2)
                    simulation_grid.add_particle(x, y, random.choice(element_keys))
                    if simulation_grid.grid[x, y]:
                        simulation_grid.grid[x, y].color = color
                simulation_grid.render()

            simulation_grid.on_resized = add_test_particles

    def open_modal(self):
        modal = PickerModal()
        modal.open()


if __name__ == "__main__":
    app = ParticleSimApp()
    app.run()
