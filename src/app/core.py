import random

from kivy.core.window import Window
from kivy.app import App
from kivy.clock import Clock
from app.ui.elements import load_elements
from app.ui.cursor import SimulationCursor
from app.ui.widgets import SimulationGrid
from app.ui.modals.PickerModal import PickerModal

ELEMENTS = load_elements()
simulation_grid = SimulationGrid()
simulation_cursor = SimulationCursor()


class ParticleSimApp(App):
    def build(self):
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        return self.root

    def update(self, dt):
        if self.root and "simulation_grid" in self.root.ids:
            self.root.ids.simulation_grid.update(dt)

    def on_start(self):
        colors = [
            [round(random.random(), 2), round(random.random(), 2), random.random(), 1]
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

    def get_mouse_pos(self):
        return Window.mouse_pos
