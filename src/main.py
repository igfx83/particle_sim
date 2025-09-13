# pyright: reportAttributeAccessIssue=false
import random
import os
from kivy.logger import Logger

from kivy.core.window import Window
from kivy.app import App
from kivy.lang import Builder
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.popup import ModalView
from services.state import AppState
from ui.elements import load_elements
from ui.modals.picker.PickerModal import PickerModal
# from kivy.utils import platform
#
# platform = platform
#
#
# def simulate_mobile():
#     return "android"
#
#
# platform = simulate_mobile()


class ParticleSimApp(App):
    fps = NumericProperty(0)
    settings = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Particle Simulation"
        self.icon = "assets/icon.png"
        self.root = None
        self.state = AppState()
        self.elements = load_elements()
        self._is_running = False
        Cache.register("resources", timeout=60)

    def build(self):
        kv_path = os.path.join(os.path.dirname(__file__), "kv", "particlesim.kv")
        root = Builder.load_file(kv_path)
        return root

    def dismiss_option_popup(self, instance, value=0):
        if value > 0:
            self._is_running = False
            instance.dismiss()
            self.state.dev_states.update({"test_gravity": f"{value}"})
            self.initialize_particles(num_particles=value)
            return value

        instance.dismiss()
        return True

    def display_settings(self, settings):
        self.settings = settings
        return True

    def update(self, dt):
        self.fps = Clock.get_fps()

        if self.simulation_grid and self._is_running:
            self.simulation_grid.update(dt)
        return True

    def on_start(self):
        try:
            if self.root is None:
                raise ValueError("Root widget is not initialized")
            self.simulation_cursor = self.root.ids.simulation_cursor
            Clock.schedule_once(lambda dt: self.initialize_particles(), 0.2)
        except Exception as e:
            Logger.error(f"Error during on_start: {e}")

    def open_option_popup(self):
        Factory.OptionPopup().open()

    def initialize_particles(self, num_particles=500):
        if self.root and self.root.ids.simulation_grid:
            self.simulation_grid = self.root.ids.simulation_grid
            element_keys = list(self.elements.keys())

            if self.simulation_grid.wait_for_ready(timeout=5):
                for _ in range(num_particles):
                    x = random.randint(0, self.simulation_grid.grid_width - 2)
                    y = random.randint(0, self.simulation_grid.grid_height - 2)
                    self.simulation_grid.add_particle(x, y, random.choice(element_keys))
                self._is_running = True
                Clock.unschedule(self.update)

                Clock.schedule_interval(self.update, 1.0 / 60.0)

    def open_modal(self):
        modal: ModalView = PickerModal()
        self._is_running = False
        modal.bind(on_pre_dismiss=lambda instance: setattr(self, "_is_running", True))
        modal.open()
        self.open_settings()

    def get_mouse_pos(self):
        return Window.mouse_pos


if __name__ == "__main__":
    app = ParticleSimApp()
    app.run()
