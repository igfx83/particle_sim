import random
import os
from kivy.logger import Logger
from kivy.core.window import Window
from kivy.app import App
from kivy.lang import Builder
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.properties import NumericProperty, ObjectProperty
from services.state import AppState
from ui.elements import load_elements
from ui.modals.PickerModal import PickerModal


class ParticleSimApp(App):
    fps = NumericProperty(0)
    settings = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Particle Simulation"
        self.icon = "assets/icon.png"
        self.root = None
        self.state = AppState()
        self.simulation_cursor = None
        self.elements = load_elements()
        self._is_running = True
        Cache.register("resources", timeout=60)

    def build(self):
        kv_path = os.path.join(os.path.dirname(__file__), "kv", "particlesim.kv")
        root = Builder.load_file(kv_path)
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        return root

    def display_settings(self, settings):
        self.settings = settings

    def update(self, dt):
        self.fps = Clock.get_fps()
        if self.root and "simulation_grid" in self.root.ids and self._is_running:
            self.root.ids.simulation_grid.update(dt)
        if random.random() < 0.01:  # 1% chance each frame
            stats = self.root.ids.simulation_grid.get_stats()
            print(
                f"Active: {stats['active_particles']}, "
                f"Burning: {stats['burning_particles']}, "
                f"Avg Temp: {stats['average_temperature']:.1f}°C"
            )

    def on_start(self):
        try:
            self.simulation_cursor = self.root.ids.simulation_cursor
            Clock.schedule_once(self.initialize_particles, 0.2)
        except Exception as e:
            Logger.error(f"Error during on_start: {e}")

    def initialize_particles(self, dt):
        if self.root and self.root.ids.simulation_grid:
            simulation_grid = self.root.ids.simulation_grid
            element_keys = list(self.elements.keys())
            if not element_keys:
                raise ValueError("No elements loaded into ELEMENTS dictionary")

            for _ in range(1000):
                x = random.randint(0, simulation_grid.spatial_grid.shape[0] - 2)
                y = random.randint(0, simulation_grid.spatial_grid.shape[1] - 2)
                simulation_grid.add_particle(x, y, random.choice(element_keys))
            simulation_grid.render()

    def open_modal(self):
        modal = PickerModal()
        modal.open()

    def get_mouse_pos(self):
        return Window.mouse_pos


if __name__ == "__main__":
    app = ParticleSimApp()
    app.run()
