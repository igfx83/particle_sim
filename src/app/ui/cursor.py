# pyright: reportAttributeAccessIssue=false
import logging
import numpy as np
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ColorProperty,
)


class SimulationCursor(Widget):
    shape = StringProperty("ellipse")
    cursor_width = NumericProperty(10)
    cursor_height = NumericProperty(10)
    selected_element = StringProperty(None, allownone=True)
    cursor_color = ColorProperty([1, 1, 1, 0.7])

    def __init__(self, **kwargs):
        super(SimulationCursor, self).__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouse_pos)
        Window.bind(on_key_down=self.on_key_down)
        Window.bind(on_motion=self.handle_motion)
        self.app = App.get_running_app()
        self.modifiers = []
        self._modal_open = False

        self.pixel_size = 4
        self.cursor_texture = None
        self.cursor_mesh = None
        self.cursor_data = None
        self.current_mouse_pos = (0, 0)

        self.bind(
            shape=self._update_cursor_texture,
            cursor_width=self._update_cursor_texture,
            cursor_height=self._update_cursor_texture,
            cursor_color=self._update_cursor_texture,
        )

        self._setup_cursor_rendering()

    def _setup_cursor_rendering(self):
        """Initialize cursor texture and mesh"""
        # Start with a reasonable size - will be updated when properties change
        self._update_cursor_texture()

    def _update_cursor_texture(self, *args):
        """Update the cursor texture based on current shape and size"""
        # Calculate texture dimensions based on cursor size
        tex_width = max(int(self.cursor_width / self.pixel_size) + 2, 4)
        tex_height = max(int(self.cursor_height / self.pixel_size) + 2, 4)

        # Create or recreate texture
        self.cursor_texture = Texture.create(size=(tex_width, tex_height))
        self.cursor_texture.mag_filter = "nearest"
        self.cursor_texture.min_filter = "nearest"

        # Create texture data array
        self.cursor_data = np.zeros((tex_height, tex_width, 4), dtype=np.uint8)

        # Generate the cursor shape
        self._generate_cursor_shape(tex_width, tex_height)

        # Create/update mesh
        self._create_cursor_mesh(tex_width, tex_height)

        # Update texture with data
        self.cursor_texture.blit_buffer(
            self.cursor_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
        )

    def _create_cursor_mesh(self, tex_width, tex_height):
        pass

    def _generate_cursor_shape(self, width, height):
        pass

    def _set_cursor_shape(self, shape: str) -> None:
        if shape == self.shape:
            return
        if shape not in ["square", "ellipse", "triangle"]:
            return

    def on_mouse_pos(self, window, pos):
        if self._modal_open:
            return

        if self.app and self.app.root:
            self.pos = (pos[0] - self.cursor_width / 2, pos[1] - self.cursor_height / 2)
            grid = self.app.root.ids.simulation_grid
            hud = self.app.root.ids.hud_label
            if grid.collide_point(*pos):
                particle = grid.get_particle_at(pos)
                if particle:
                    hud.text = (
                        f"Element: {particle.id}\n"
                        f"Temperature: {particle.temperature:.1f}°C\n"
                        f"State: {particle.state}"
                    )
                else:
                    hud.text = ""
            else:
                hud.text = ""

    def on_key_up(self, window, key, scancode):
        # Reset modifiers when key is released
        self.modifiers = []

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        self.modifiers = modifier
        if codepoint == "tab":
            self.shape = "square"
        if codepoint == "tab":
            self.shape = "ellipse"
        elif codepoint == "t":
            self.shape = "triangle"
        if "ctrl" in modifier and codepoint == "w":
            self.cursor_width = min(self.cursor_width + 2, 50)
        elif "ctrl" in modifier and codepoint == "s":
            self.cursor_width = max(self.cursor_width - 2, 5)
        elif "shift" in modifier and codepoint == "w":
            self.cursor_height = min(self.cursor_height + 2, 50)
        elif "shift" in modifier and codepoint == "s":
            self.cursor_height = max(self.cursor_height - 2, 5)
        elif codepoint == "w":
            self.cursor_width = self.cursor_height = min(self.cursor_width + 2, 50)
        elif codepoint == "s":
            self.cursor_width = self.cursor_height = max(self.cursor_width - 2, 5)

    def handle_motion(self, window, etype, me):
        if etype == "scroll":
            if "ctrl" in me.modifiers:
                self.cursor_width = max(min(self.cursor_width + me.dz * 2, 50), 5)
            elif "shift" in me.modifiers:
                self.cursor_height = max(min(self.cursor_height + me.dz * 2, 50), 5)
            else:
                self.cursor_width = self.cursor_height = max(
                    min(self.cursor_width + me.dz * 2, 50), 5
                )

    def on_grid_touch_down(self, grid, touch):
        if grid.collide_point(*touch.pos):
            if touch.button == "left" and self.selected_element:
                grid.place_particles(
                    touch.pos,
                    self.shape,
                    self.cursor_width,
                    self.cursor_height,
                    self.selected_element,
                )
            elif touch.button == "middle" or (
                "alt" in self.modifiers and touch.button == "left"
            ):
                particle = grid.get_particle_at(touch.pos)
                if particle:
                    self.selected_element = particle.id
                    logging.debug(f"Sampled element: {self.selected_element}")

    def on_grid_touch_move(self, grid, touch):
        if (
            grid.collide_point(*touch.pos)
            and touch.button == "left"
            and self.selected_element
        ):
            grid.place_particles(
                touch.pos,
                self.shape,
                self.cursor_width,
                self.cursor_height,
                self.selected_element,
            )


Factory.register("SimulationCursor", cls=SimulationCursor)
