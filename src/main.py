# pyright: reportAttributeAccessIssue=false
import logging
import os
import random
import time

from kivy.app import App
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import ModalView

from services.state import AppState
from ui.elements import load_elements
from ui.modals.picker.PickerModal import PickerModal
from ui.widgets import SimulationGrid

logging.getLogger("numba").setLevel("WARNING")


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
        self.simulation_grid: SimulationGrid | None = None
        Cache.register("resources", timeout=60)

    def build(self):
        kv_path = os.path.join(os.path.dirname(__file__), "kv", "particlesim.kv")
        root = Builder.load_file(kv_path)
        return root

    def open_gravity_dropdown(self, button):
        """Create and open the gravity dropdown"""
        # Create dropdown
        dropdown = DropDown()

        # Vertical option
        btn_vertical = Button(
            text="Vertical", size_hint_y=None, height=dp(40), font_size=16
        )
        btn_vertical.bind(
            on_release=lambda x: self._select_gravity_mode(
                0, "Vertical", dropdown, button
            )
        )
        dropdown.add_widget(btn_vertical)

        # Loop option
        btn_loop = Button(text="Loop", size_hint_y=None, height=dp(40), font_size=16)
        btn_loop.bind(
            on_release=lambda x: self._select_gravity_mode(1, "Loop", dropdown, button)
        )
        dropdown.add_widget(btn_loop)

        # Void option
        btn_void = Button(text="Void", size_hint_y=None, height=dp(40), font_size=16)
        btn_void.bind(
            on_release=lambda x: self._select_gravity_mode(2, "Void", dropdown, button)
        )
        dropdown.add_widget(btn_void)

        # Show dropdown
        dropdown.open(button)

    def _select_gravity_mode(self, mode, text, dropdown, button):
        """Handle gravity mode selection"""
        # Update button text
        button.text = text
        if self.simulation_grid:
            # Update simulation grid
            self.simulation_grid.switch_edge_mode(mode)

        # Close dropdown
        dropdown.dismiss()

        print(f"Gravity mode set to: {mode}")  # Debug

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
        # Need to fix this method

        self.fps = Clock.get_fps()

        if self.simulation_grid and self._is_running:
            self.simulation_grid.update(dt)
        return True

    def on_start(self):
        try:
            if self.root is None:
                raise ValueError("Root widget is not initialized")
            self.simulation_cursor = self.root.ids.simulation_cursor
            self.simulation_grid = self.root.ids.simulation_grid
            Clock.schedule_once(lambda dt: self.initialize_particles(), 0.2)

            Clock.schedule_interval(self.update, 1.0 / 60.0)
        except Exception as e:
            Logger.error(f"Error during on_start: {e}")

    def open_option_popup(self):
        Factory.OptionPopup().open()

    def initialize_particles(self, num_particles=500):
        element_keys = list(self.elements.keys())

        if self.simulation_grid and self.simulation_grid.wait_for_ready(timeout=5):
            for _ in range(num_particles):
                x = random.randint(0, self.simulation_grid.grid_width - 2)
                y = random.randint(0, self.simulation_grid.grid_height - 2)
                self.simulation_grid.add_particle(x, y, random.choice(element_keys))
            time.sleep(1)
            self._is_running = True

    def open_modal(self):
        modal: ModalView = PickerModal()
        self._is_running = False
        modal.bind(on_pre_dismiss=lambda instance: setattr(self, "_is_running", True))
        modal.open()
        self.open_settings()


if __name__ == "__main__":
    app = ParticleSimApp()
    app.run()
