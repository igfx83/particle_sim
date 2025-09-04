# pyright: reportAttributeAccessIssue=false
import numpy as np
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import Color, Mesh, PushMatrix, PopMatrix, Translate
from kivy.graphics.texture import Texture

# from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ColorProperty,
    ObjectProperty,
)

STATES = ["Solid", "Liquid", "Gas"]


class SimulationCursor(Widget):
    simulation_grid = ObjectProperty(None, allownone=True)
    hud = ObjectProperty(None, allownone=True)
    shape = StringProperty("ellipse")
    cursor_width = NumericProperty(10)
    cursor_height = NumericProperty(10)
    cursor_data = ObjectProperty(None, allownone=True)
    selected_element = StringProperty(None, allownone=True)
    cursor_color = ColorProperty([1, 1, 1, 0.7])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = App.get_running_app()
        self.pixel_size = 4
        self.cursor_texture = None
        self.cursor_mesh = None
        self.current_pos = (0, 0)
        self.modifiers = []
        self._modal_open = False

        # Bind property changes to update cursor
        self.bind(
            shape=self.update_cursor,
            cursor_width=self.update_cursor,
            cursor_height=self.update_cursor,
            cursor_color=self.update_cursor,
        )

        # Initialize cursor
        self.update_cursor()

    def update_cursor(self, *args):
        """Update cursor texture and mesh when properties change."""
        # Calculate texture dimensions
        tex_width = max(int(self.cursor_width / self.pixel_size) + 2, 4)
        tex_height = max(int(self.cursor_height / self.pixel_size) + 2, 4)

        # Create or update texture
        self.cursor_texture = Texture.create(size=(tex_width, tex_height))
        self.cursor_texture.mag_filter = "nearest"
        self.cursor_texture.min_filter = "nearest"

        # Generate texture data
        self.cursor_data = np.zeros((tex_height, tex_width, 4), dtype=np.uint8)
        center_x, center_y = tex_width // 2, tex_height // 2
        half_width = self.cursor_width / self.pixel_size / 2
        half_height = self.cursor_height / self.pixel_size / 2
        color = [int(c * 255) for c in self.cursor_color]

        # Generate shape
        for y in range(tex_height):
            for x in range(tex_width):
                if self._point_on_perimeter(
                    x, y, center_x, center_y, half_width, half_height
                ):
                    self.cursor_data[y, x] = color

        # Update texture
        self.cursor_texture.blit_buffer(
            self.cursor_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
        )

        # Create mesh
        mesh_width = tex_width * self.pixel_size
        mesh_height = tex_height * self.pixel_size
        vertices = [
            0,
            0,
            0,
            0,  # bottom-left
            mesh_width,
            0,
            1,
            0,  # bottom-right
            mesh_width,
            mesh_height,
            1,
            1,  # top-right
            0,
            mesh_height,
            0,
            1,  # top-left
        ]
        indices = [0, 1, 2, 2, 3, 0]
        self.cursor_mesh = Mesh(vertices=vertices, indices=indices, mode="triangles")
        self.cursor_mesh.texture = self.cursor_texture

        # Trigger redraw
        self.render_cursor()

    def _point_in_shape(self, x, y, center_x, center_y, half_width, half_height):
        """Check if a point is inside the cursor shape."""
        dx = x - center_x
        dy = y - center_y

        if self.shape == "ellipse":
            return (
                half_width > 0
                and half_height > 0
                and (dx / half_width) ** 2 + (dy / half_height) ** 2 <= 1
            )
        elif self.shape == "square":
            return abs(dx) <= half_width and abs(dy) <= half_height
        elif self.shape == "triangle":
            if dy < -half_height or dy > half_height:
                return False
            y_normalized = (dy + half_height) / (2 * half_height)
            width_at_y = half_width * (1 - y_normalized)
            return abs(dx) <= width_at_y
        return False

    def get_cursor_particle_positions(self, world_pos):
        """Get grid positions for particle placement."""
        grid_x = int(
            (world_pos[0] - self.simulation_grid.x) / self.simulation_grid.pixel_size
        )
        grid_y = int(
            (world_pos[1] - self.simulation_grid.y) / self.simulation_grid.pixel_size
        )
        half_w = int(self.cursor_width / self.simulation_grid.pixel_size / 2)
        half_h = int(self.cursor_height / self.simulation_grid.pixel_size / 2)
        positions = []

        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                x, y = grid_x + dx, grid_y + dy
                if (
                    0 <= x < self.simulation_grid.grid_width
                    and 0 <= y < self.simulation_grid.grid_height
                    and self._is_in_cursor_shape(dx, dy, half_w, half_h)
                ):
                    positions.append((x, y))

        return positions

    def _point_on_perimeter(self, x, y, center_x, center_y, half_width, half_height):
        """Check if a point is on the perimeter of the cursor shape."""
        dx = x - center_x
        dy = y - center_y

        if self.shape == "ellipse":
            # Perimeter: distance from center is close to 1 (within a tolerance)
            r = (dx / half_width) ** 2 + (dy / half_height) ** 2
            return abs(r - 1) < 0.15
        elif self.shape == "square":
            # Perimeter: on the edge of the square
            return (
                (abs(dx) == half_width or abs(dy) == half_height)
                and abs(dx) <= half_width
                and abs(dy) <= half_height
            )
        elif self.shape == "triangle":
            # Perimeter: on one of the triangle's edges
            if dy < -half_height or dy > half_height:
                return False
            y_normalized = (dy + half_height) / (2 * half_height)
            width_at_y = half_width * (1 - y_normalized)
            return abs(dx) == round(width_at_y)
        return False

    def on_touch_down(self, touch):
        """Handle touch/mouse down events."""
        if self._modal_open:
            return

        if not self.simulation_grid.collide_point(*touch.pos):
            return

        # if touch.button == "left" and self.selected_element:
        #     self.simulationg_grid.place_particles(
        #         touch.pos,
        #         self.shape,
        #         self.cursor_width,
        #         self.cursor_height,
        #         self.selected_element,
        #     )
        # elif touch.button == "middle" or (
        #     "alt" in self.modifiers and touch.button == "left"
        # ):
        #     particle = self.simulation_grid.get_particle_at(*touch.pos)
        #     if particle:
        #         self.selected_element = particle["element_id"]
        # if hasattr(particle, "color"):
        #    qqqqqqqqqq self.cursor_color = [*particle.color[:3], 0.7]
        # Logger.debug(f"Sampled element: {self.selected_element}")

    def on_touch_move(self, touch):
        """Handle touch/mouse move events."""
        if self._modal_open:
            return

        self.current_pos = touch.pos
        self._update_position_and_hud(touch.pos)

        if (
            self.simulation_grid
            and self.simulation_grid.collide_point(*touch.pos)
            and touch.button == "left"
            and self.selected_element
        ):
            self.simulation_grid.place_particles(
                touch.pos,
                self.shape,
                self.cursor_width,
                self.cursor_height,
                self.selected_element,
            )

    def on_touch_up(self, touch):
        """Handle touch/mouse up events."""
        self.modifiers = []

    def on_scroll(self, window, x, y, dx, dy):
        """Handle scroll events."""
        if self._modal_open:
            return

        delta = dy * 2
        if "ctrl" in self.modifiers:
            self.cursor_width = max(min(self.cursor_width + delta, 100), 2)
        elif "shift" in self.modifiers:
            self.cursor_height = max(min(self.cursor_height + delta, 100), 2)
        else:
            self.cursor_width = self.cursor_height = max(
                min(self.cursor_width + delta, 100), 2
            )

    def _update_position_and_hud(self, pos):
        """Update cursor position and HUD information."""
        cursor_pixel_width = (self.cursor_width / self.pixel_size + 2) * self.pixel_size
        cursor_pixel_height = (
            self.cursor_height / self.pixel_size + 2
        ) * self.pixel_size
        self.pos = (pos[0] - cursor_pixel_width / 2, pos[1] - cursor_pixel_height / 2)

        if not self.simulation_grid or not self.hud:
            return

        if self.simulation_grid.collide_point(*pos):
            particle_data = self.simulation_grid.get_particle_at(*pos)
            if isinstance(particle_data, np.void):
                element_map = {
                    v: k for k, v in self.simulation_grid.element_map.items()
                }
                self.hud.text = (
                    f"Element: {element_map[particle_data['element_id']]}\n"
                    f"Temperature: {particle_data['temperature']:.1f}°C\n"
                    f"State: {STATES[particle_data['state']]}"
                )
            else:
                self.hud.text = f"Cursor: {self.shape} ({int(self.cursor_width)}x{int(self.cursor_height)})"
                if self.selected_element:
                    self.hud.text += f"\nSelected: {self.selected_element}"
        else:
            self.hud.text = ""

    def render_cursor(self):
        """Render the cursor on screen."""
        if not self.cursor_mesh or not self.cursor_texture or not self.canvas:
            return

        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1)
            PushMatrix()
            Translate(self.x, self.y)
            self.canvas.add(self.cursor_mesh)
            PopMatrix()

    def on_pos(self, *args):
        """Redraw when position changes."""
        self.render_cursor()

    def on_selected_element(self, instance, value):
        """Update cursor color when selected element changes."""
        if not value:
            self.cursor_color = [1.0, 1.0, 1.0, 0.7]
            return

        try:
            from ui.elements import load_elements

            elements = load_elements()
            if value in elements:
                color = elements[value]["intrinsic_properties"]["color"]
                self.cursor_color = [*color[:3], 0.7]
        except (ValueError, KeyError):
            self.cursor_color = [1.0, 1.0, 1.0, 0.7]

    def on_keyboard(self, *kwargs):
        """Handle keyboard input."""
        # Shape selection
        key, scancode, codepoint, modifier = kwargs

        shape_map = {"q": "square", "e": "ellipse", "r": "triangle"}
        if codepoint in shape_map:
            self.shape = shape_map[codepoint]

        # Size adjustments
        if codepoint in ("w", "s"):
            delta = 2 if codepoint == "w" else -2
            if "ctrl" in modifier:
                self.cursor_width = max(min(self.cursor_width + delta, 100), 2)
            elif "shift" in modifier:
                self.cursor_height = max(min(self.cursor_height + delta, 100), 2)
            else:
                self.cursor_width = self.cursor_height = max(
                    min(self.cursor_width + delta, 100), 2
                )


# Bind keyboard and scroll events at the class level
Window.bind(on_keyboard=SimulationCursor.on_keyboard.__get__(SimulationCursor))
Window.bind(on_scroll=SimulationCursor.on_scroll)

Factory.register("SimulationCursor", cls=SimulationCursor)
