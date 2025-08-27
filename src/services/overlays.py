import numpy as np
from kivy.graphics import PushMatrix, PopMatrix, Scale, Translate
from kivy.graphics.texture import Texture
from kivy.graphics import Mesh


class AirPropertyOverlay:
    """Ultra-optimized air property overlay system"""

    def __init__(self, simulation_grid):
        self.sim_grid = simulation_grid
        self.width = simulation_grid.grid_width
        self.height = simulation_grid.grid_height

        # Simple overlay configuration
        self.layers = {
            "temperature": {"enabled": False, "opacity": 0.6},
            "velocity": {"enabled": False, "opacity": 0.7},
        }

        # Pre-computed colormaps - cache as class variables for memory efficiency
        if not hasattr(AirPropertyOverlay, "_thermal_colormap"):
            AirPropertyOverlay._thermal_colormap = self._precompute_thermal_colormap()
            AirPropertyOverlay._velocity_colormap = self._precompute_velocity_colormap()

        self.thermal_colormap = AirPropertyOverlay._thermal_colormap
        self.velocity_colormap = AirPropertyOverlay._velocity_colormap

        # Single overlay texture and data array - preallocated
        self.overlay_texture = None
        self.overlay_mesh = None
        self.overlay_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Simplified air grid - contiguous memory layout for better cache performance
        air_shape = (self.width, self.height)
        self.air_temp = np.full(air_shape, 20.0, dtype=np.float32, order="C")
        self.air_vel_x = np.zeros(air_shape, dtype=np.float32, order="C")
        self.air_vel_y = np.zeros(air_shape, dtype=np.float32, order="C")

        # Pre-allocate working arrays to avoid repeated allocations
        self._temp_colors = np.empty((self.width, self.height, 4), dtype=np.uint8)
        self._temp_indices = np.empty((self.width, self.height), dtype=np.uint8)
        self._temp_norm = np.empty((self.width, self.height), dtype=np.float32)

        # Update counters
        self.update_counter = 0
        self.air_update_interval = 8  # Increased from 5 for better performance
        self.render_skip_counter = 0
        self.render_interval = 2  # Only render overlay every 2 frames

        self._setup_overlay_rendering()

    @staticmethod
    def _precompute_thermal_colormap():
        """Pre-compute thermal colormap lookup table - static for memory sharing"""
        colormap = np.empty((256, 4), dtype=np.uint8)

        # Vectorized colormap computation
        vals = np.linspace(0, 1, 256)

        # Black to red phase (0-0.33)
        mask1 = vals < 0.33
        t1 = vals[mask1] / 0.33
        colormap[mask1, 0] = (255 * t1).astype(np.uint8)
        colormap[mask1, 1] = 0
        colormap[mask1, 2] = 0
        colormap[mask1, 3] = 255

        # Red to yellow phase (0.33-0.66)
        mask2 = (vals >= 0.33) & (vals < 0.66)
        t2 = (vals[mask2] - 0.33) / 0.33
        colormap[mask2, 0] = 255
        colormap[mask2, 1] = (255 * t2).astype(np.uint8)
        colormap[mask2, 2] = 0
        colormap[mask2, 3] = 255

        # Yellow to white phase (0.66-1.0)
        mask3 = vals >= 0.66
        t3 = (vals[mask3] - 0.66) / 0.34
        colormap[mask3, 0] = 255
        colormap[mask3, 1] = 255
        colormap[mask3, 2] = (255 * t3).astype(np.uint8)
        colormap[mask3, 3] = 255

        return colormap

    @staticmethod
    def _precompute_velocity_colormap():
        """Pre-compute velocity colormap - static for memory sharing"""
        colormap = np.empty((256, 4), dtype=np.uint8)

        vals = np.linspace(0, 1, 256)
        colormap[:, 0] = (255 * vals).astype(np.uint8)  # Red channel
        colormap[:, 1] = 0  # Green channel
        colormap[:, 2] = (255 * (1 - vals)).astype(np.uint8)  # Blue channel
        colormap[:, 3] = 255  # Alpha channel

        return colormap

    def _setup_overlay_rendering(self):
        """Initialize overlay texture and mesh"""
        self.overlay_texture = Texture.create(size=(self.width, self.height))
        self.overlay_texture.mag_filter = "nearest"
        self.overlay_texture.min_filter = "nearest"

        # Simple quad mesh - reuse vertices array
        vertices = np.array(
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.float32
        )
        indices = [0, 1, 2, 2, 3, 0]
        self.overlay_mesh = Mesh(vertices=vertices, indices=indices, mode="triangles")

    def toggle_layer(self, layer_name):
        """Toggle layer visibility"""
        if layer_name in self.layers:
            self.layers[layer_name]["enabled"] = not self.layers[layer_name]["enabled"]
            return self.layers[layer_name]["enabled"]
        return False

    def update_air_simple(self, dt):
        """Ultra-simplified air update with minimal allocations"""
        self.update_counter += 1

        # Skip air updates more frequently
        if self.update_counter % self.air_update_interval != 0:
            return

        # In-place operations to avoid allocations
        np.multiply(self.air_temp, 0.98, out=self.air_temp)  # Cooling
        np.multiply(self.air_vel_x, 0.9, out=self.air_vel_x)  # Velocity decay
        np.multiply(self.air_vel_y, 0.9, out=self.air_vel_y)

        # Early exit if no particles
        if self.sim_grid.particle_count == 0:
            return

        # Get active particles - avoid intermediate array creation
        active_count = np.sum(self.sim_grid.active[: self.sim_grid.particle_count])
        if active_count == 0:
            return

        # Only process a subset of hot particles for better performance
        hot_particle_limit = min(100, active_count)  # Limit processing

        active_indices = np.where(self.sim_grid.active[: self.sim_grid.particle_count])[
            0
        ]

        # Sample hot particles if too many
        if len(active_indices) > hot_particle_limit:
            # Get hottest particles first
            temps = self.sim_grid.temperatures[active_indices]
            hot_mask = temps > 50
            hot_indices = active_indices[hot_mask]

            if len(hot_indices) > hot_particle_limit:
                # Sample randomly from hot particles
                sample_indices = np.random.choice(
                    hot_indices, hot_particle_limit, replace=False
                )
                active_indices = sample_indices
            else:
                active_indices = hot_indices

        if len(active_indices) == 0:
            return

        # Vectorized air heating - bounds check first
        xs = self.sim_grid.x_coords[active_indices]
        ys = self.sim_grid.y_coords[active_indices]
        temps = self.sim_grid.temperatures[active_indices]
        burning = self.sim_grid.burning[active_indices]

        # Single bounds and temperature check
        valid = (
            (xs < self.width)
            & (ys < self.height)
            & (xs >= 0)
            & (ys >= 0)
            & (temps > 50)
        )

        if not np.any(valid):
            return

        # Apply to valid particles only
        valid_x = xs[valid]
        valid_y = ys[valid]
        valid_temps = temps[valid]
        valid_burning = burning[valid]

        # Vectorized air heating - use np.maximum for in-place operation
        heat_values = valid_temps * 0.5
        np.maximum(
            self.air_temp[valid_x, valid_y],
            heat_values,
            out=self.air_temp[valid_x, valid_y],
        )

        # Add upward velocity for burning particles - vectorized
        if np.any(valid_burning):
            burn_x = valid_x[valid_burning]
            burn_y = valid_y[valid_burning]
            self.air_vel_y[burn_x, burn_y] += 2.0

    def render_overlay(self, canvas):
        """Optimized overlay rendering with frame skipping"""
        if not any(layer["enabled"] for layer in self.layers.values()):
            return

        # Skip rendering frames for better performance
        self.render_skip_counter += 1
        if self.render_skip_counter % self.render_interval != 0:
            return

        # Clear overlay efficiently
        self.overlay_data.fill(0)

        # Render enabled layers
        if self.layers["temperature"]["enabled"]:
            self._render_temperature_optimized()

        if self.layers["velocity"]["enabled"]:
            self._render_velocity_optimized()

        # Single texture upload with error handling
        try:
            self.overlay_texture.blit_buffer(
                self.overlay_data.tobytes(), colorfmt="rgba", bufferfmt="ubyte"
            )
        except Exception as e:
            return

        # Render mesh
        with canvas:
            PushMatrix()
            Translate(self.sim_grid.x, self.sim_grid.y)
            Scale(
                self.sim_grid.pixel_size * self.width,
                self.sim_grid.pixel_size * self.height,
                1,
            )

            self.overlay_mesh.texture = self.overlay_texture
            canvas.add(self.overlay_mesh)

            PopMatrix()

    def _render_temperature_optimized(self):
        """Optimized temperature rendering using pre-allocated arrays"""
        # Use pre-allocated array for normalization
        np.clip((self.air_temp - 20.0) * (1.0 / 180.0), 0.0, 1.0, out=self._temp_norm)

        # Convert to indices in-place
        np.multiply(self._temp_norm, 255, out=self._temp_norm)
        self._temp_norm.astype(np.uint8, out=self._temp_indices)

        # Vectorized colormap lookup - use advanced indexing
        self._temp_colors[:, :] = self.thermal_colormap[self._temp_indices]

        # Apply opacity in-place
        opacity_uint8 = int(self.layers["temperature"]["opacity"] * 255)
        self._temp_colors[:, :, 3] = opacity_uint8

        # Transpose and copy - minimize memory operations
        colors_t = np.transpose(self._temp_colors, (1, 0, 2))

        # Simple overwrite for maximum performance
        mask = colors_t[:, :, 3] > 0
        self.overlay_data[mask] = colors_t[mask]

    def _render_velocity_optimized(self):
        """Optimized velocity rendering"""
        # Calculate magnitude in-place using pre-allocated array
        np.multiply(self.air_vel_x, self.air_vel_x, out=self._temp_norm)
        self._temp_norm += self.air_vel_y * self.air_vel_y
        np.sqrt(self._temp_norm, out=self._temp_norm)

        # Normalize to 0-1 range in-place
        np.clip(self._temp_norm * 0.2, 0.0, 1.0, out=self._temp_norm)  # 0.2 = 1/5

        # Convert to indices
        np.multiply(self._temp_norm, 255, out=self._temp_norm)
        self._temp_indices = self._temp_norm.astype(np.uint8)

        # Colormap lookup
        self._temp_colors[:, :] = self.velocity_colormap[self._temp_indices]

        # Apply opacity
        opacity_uint8 = int(self.layers["velocity"]["opacity"] * 255)
        self._temp_colors[:, :, 3] = opacity_uint8

        # Transpose
        colors_t = np.transpose(self._temp_colors, (1, 0, 2))

        # Alpha blending - simplified for performance
        mask = colors_t[:, :, 3] > 0
        alpha_factor = self.layers["velocity"]["opacity"]

        # Simple blend operation
        self.overlay_data[mask, :3] = (
            self.overlay_data[mask, :3] * (1 - alpha_factor)
            + colors_t[mask, :3] * alpha_factor
        ).astype(np.uint8)

        # Set alpha
        self.overlay_data[mask, 3] = np.maximum(
            self.overlay_data[mask, 3], colors_t[mask, 3]
        )

    def resize(self, new_width, new_height):
        """Optimized resize with minimal allocations"""
        if new_width == self.width and new_height == self.height:
            return

        self.width = new_width
        self.height = new_height

        # Resize arrays efficiently
        air_shape = (new_width, new_height)
        self.air_temp = np.full(air_shape, 20.0, dtype=np.float32, order="C")
        self.air_vel_x = np.zeros(air_shape, dtype=np.float32, order="C")
        self.air_vel_y = np.zeros(air_shape, dtype=np.float32, order="C")
        self.overlay_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # Resize working arrays
        self._temp_colors = np.empty((new_width, new_height, 4), dtype=np.uint8)
        self._temp_indices = np.empty((new_width, new_height), dtype=np.uint8)
        self._temp_norm = np.empty((new_width, new_height), dtype=np.float32)

        # Recreate texture
        self._setup_overlay_rendering()


# Optimized integration function
def integrate_overlay_with_simulation(simulation_grid):
    """Ultra-lightweight integration"""

    simulation_grid.air_overlay = AirPropertyOverlay(simulation_grid)

    # Cache original methods to avoid attribute lookups
    original_update = simulation_grid.update
    original_render = simulation_grid.render
    original_resize = simulation_grid.resize

    def enhanced_update(dt):
        original_update(dt)
        simulation_grid.air_overlay.update_air_simple(dt)

    def enhanced_render():
        original_render()
        simulation_grid.air_overlay.render_overlay(simulation_grid.canvas)

    def enhanced_resize(new_width, new_height):
        original_resize(new_width, new_height)
        simulation_grid.air_overlay.resize(new_width, new_height)

    # Replace methods
    simulation_grid.update = enhanced_update
    simulation_grid.render = enhanced_render
    simulation_grid.resize = enhanced_resize

    # Add toggle methods
    simulation_grid.toggle_temperature = (
        lambda: simulation_grid.air_overlay.toggle_layer("temperature")
    )
    simulation_grid.toggle_velocity = lambda: simulation_grid.air_overlay.toggle_layer(
        "velocity"
    )

    return simulation_grid
