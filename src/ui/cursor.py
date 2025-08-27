# pyright: reportAttributeAccessIssue=false
import logging
import numpy as np
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import Color, Mesh, PushMatrix, PopMatrix, Translate
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
        """Create mesh for cursor rendering"""
        # Calculate actual size in pixels
        mesh_width = tex_width * self.pixel_size
        mesh_height = tex_height * self.pixel_size

        # Create vertices for cursor quad
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

    def _generate_cursor_shape(self, tex_width, tex_height):
        """Generate the cursor shape in the texture data"""
        # Clear texture
        self.cursor_data.fill(0)

        # Calculate shape parameters
        center_x = tex_width // 2
        center_y = tex_height // 2
        half_width = self.cursor_width / self.pixel_size / 2
        half_height = self.cursor_height / self.pixel_size / 2

        # Convert color to uint8
        color = [
            int(self.cursor_color[0] * 255),
            int(self.cursor_color[1] * 255),
            int(self.cursor_color[2] * 255),
            int(self.cursor_color[3] * 255),
        ]

        # Generate shape based on cursor shape property
        for y in range(tex_height):
            for x in range(tex_width):
                if self._point_in_cursor_shape(
                    x, y, center_x, center_y, half_width, half_height
                ):
                    self.cursor_data[y, x] = color

    def _point_in_cursor_shape(self, x, y, center_x, center_y, half_width, half_height):
        """Check if a point is inside the cursor shape"""
        dx = x - center_x
        dy = y - center_y

        if self.shape == "ellipse":
            # Ellipse equation: (x/a)² + (y/b)² <= 1
            if half_width <= 0 or half_height <= 0:
                return False
            return (dx / half_width) ** 2 + (dy / half_height) ** 2 <= 1

        elif self.shape == "square":
            # Rectangle bounds
            return abs(dx) <= half_width and abs(dy) <= half_height

        elif self.shape == "triangle":
            # Triangle pointing upward
            # Base of triangle is at bottom, tip at top
            if dy > half_height or dy < -half_height:
                return False

            # Calculate triangle bounds at this y level
            # At y = -half_height (bottom), width = full width
            # At y = half_height (top), width = 0
            y_normalized = (dy + half_height) / (
                2 * half_height
            )  # 0 at bottom, 1 at top
            width_at_y = half_width * (1 - y_normalized)

            return abs(dx) <= width_at_y

        return False

    def _is_in_cursor_shape(self, dx, dy, half_w, half_h):
        """Check if offset from center is within cursor shape (matches SimulationGrid logic)"""
        if self.shape == "square":
            return True  # Already filtered by range
        elif self.shape == "ellipse":
            if half_w <= 0 or half_h <= 0:
                return dx == 0 and dy == 0
            norm_x = dx / half_w
            norm_y = dy / half_h
            return norm_x * norm_x + norm_y * norm_y <= 1
        elif self.shape == "triangle":
            if dy < -half_h or dy > half_h:
                return False
            # Triangle bounds
            y_rel = dy
            x_rel = abs(dx)
            return (
                y_rel >= -half_h
                and y_rel <= half_h
                and x_rel <= (half_h - y_rel) * half_w / half_h
                if half_h > 0
                else False
            )
        return True

    def get_cursor_particle_positions(self, world_pos):
        """Get the grid positions where particles would be placed for the current cursor"""
        if not self.app or not self.app.root:
            return []

        grid = self.app.root.ids.simulation_grid
        if not grid:
            return []

        # Convert world position to grid position
        grid_x = int((world_pos[0] - grid.x) / grid.pixel_size)
        grid_y = int((world_pos[1] - grid.y) / grid.pixel_size)

        # Calculate cursor bounds in grid coordinates
        half_w = int(self.cursor_width / grid.pixel_size / 2)
        half_h = int(self.cursor_height / grid.pixel_size / 2)

        positions = []

        # Generate positions based on shape
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                x, y = grid_x + dx, grid_y + dy

                # Check if position is within bounds and shape
                if (
                    0 <= x < grid.grid.shape[0]
                    and 0 <= y < grid.grid.shape[1]
                    and self._is_in_cursor_shape(dx, dy, half_w, half_h)
                ):
                    positions.append((x, y))

        return positions

    def _set_cursor_shape(self, shape: str) -> None:
        if shape == self.shape:
            return
        if shape not in ["square", "ellipse", "triangle"]:
            return

    def on_mouse_pos(self, window, pos):
        if self._modal_open:
            return

        self.current_mouse_pos = pos

        if self.app and self.app.root:
            # Update cursor position (centered on mouse)
            cursor_pixel_width = (
                self.cursor_width / self.pixel_size + 2
            ) * self.pixel_size
            cursor_pixel_height = (
                self.cursor_height / self.pixel_size + 2
            ) * self.pixel_size
            self.pos = (
                pos[0] - cursor_pixel_width / 2,
                pos[1] - cursor_pixel_height / 2,
            )

            # Update HUD with particle info
            grid = self.app.root.ids.simulation_grid
            hud = self.app.root.ids.hud_label
            if grid.collide_point(*pos):
                particle_idx = grid.get_particle_at(pos)
                if particle_idx != -1:
                    particle_data = grid.get_particle_data(particle_idx)
                    # Use particle_data instead of particle object
                else:
                    particle_data = None
                if particle_data:
                    hud.text = (
                        f"Element: {particle_data['element_name']}\n"
                        f"Temperature: {particle_data['temperature']:.1f}°C\n"
                        f"State: {particle_data['state']}"
                    )
                else:
                    hud.text = f"Cursor: {self.shape} ({int(self.cursor_width)}x{int(self.cursor_height)})"
                    if self.selected_element:
                        hud.text += f"\nSelected: {self.selected_element}"
            else:
                hud.text = ""

    def on_key_up(self, window, key, scancode):
        # Reset modifiers when key is released
        self.modifiers = []

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        self.modifiers = modifier

        # Shape selection
        if codepoint == "q":
            self.shape = "square"
        elif codepoint == "e":
            self.shape = "ellipse"
        elif codepoint == "r":
            self.shape = "triangle"

        # Size adjustments
        if "ctrl" in modifier and codepoint == "w":
            self.cursor_width = min(self.cursor_width + 2, 100)
        elif "ctrl" in modifier and codepoint == "s":
            self.cursor_width = max(self.cursor_width - 2, 2)
        elif "shift" in modifier and codepoint == "w":
            self.cursor_height = min(self.cursor_height + 2, 100)
        elif "shift" in modifier and codepoint == "s":
            self.cursor_height = max(self.cursor_height - 2, 2)
        elif codepoint == "w":
            self.cursor_width = self.cursor_height = min(self.cursor_width + 2, 100)
        elif codepoint == "s":
            self.cursor_width = self.cursor_height = max(self.cursor_width - 2, 2)

    def handle_motion(self, window, etype, me):
        if etype == "scroll":
            if "ctrl" in me.modifiers:
                self.cursor_width = max(min(self.cursor_width + me.dz * 2, 100), 2)
            elif "shift" in me.modifiers:
                self.cursor_height = max(min(self.cursor_height + me.dz * 2, 100), 2)
            else:
                self.cursor_width = self.cursor_height = max(
                    min(self.cursor_width + me.dz * 2, 100), 2
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
                    # Update cursor color to match selected element
                    if hasattr(particle, "color"):
                        self.cursor_color = [
                            particle.color[0],
                            particle.color[1],
                            particle.color[2],
                            0.7,  # Keep transparency
                        ]
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

    def render_cursor(self):
        """Render the cursor on screen"""
        if not self.cursor_mesh or not self.cursor_texture:
            return

        # Clear and redraw
        self.canvas.clear()

        with self.canvas:
            Color(1, 1, 1, 1)  # Full white tint (texture already has the color)
            PushMatrix()

            # Position the cursor
            Translate(self.x, self.y)

            # Render the textured mesh
            self.canvas.add(self.cursor_mesh)

            PopMatrix()

    def on_pos(self, *args):
        """Called when cursor position changes"""
        # Trigger a redraw
        self.render_cursor()

    def on_selected_element(self, instance, value):
        """Update cursor appearance when selected element changes"""
        if value and self.app and self.app.root:
            # Try to get element color from ELEMENTS data
            try:
                from ui.elements import load_elements

                elements = load_elements()
                if value in elements:
                    element_color = elements[value]["intrinsic_properties"]["color"]
                    self.cursor_color = [
                        element_color[0],
                        element_color[1],
                        element_color[2],
                        0.7,  # Semi-transparent
                    ]
                    return
            except ValueError:
                pass

            # Fallback to white if we can't get element color
            self.cursor_color = [1.0, 1.0, 1.0, 0.7]


Factory.register("SimulationCursor", cls=SimulationCursor)
