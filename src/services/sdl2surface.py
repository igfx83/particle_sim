import time
import ctypes
import gc
import threading
from typing import Callable
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty
from kivy.logger import Logger

try:
    import sdl2
    import sdl2.ext
    import termuxgui as tg

    SDL2_AVAILABLE = True
except ImportError:
    SDL2_AVAILABLE = False
    Logger.warning("SDL2 or termuxgui not available, falling back to Kivy rendering")


class NativeSurfaceWidget(Widget):
    """
    A Kivy widget that renders particles using native SDL2 surface for performance
    """

    # Properties
    surface_width = NumericProperty(500)
    surface_height = NumericProperty(500)
    fps_target = NumericProperty(60)
    is_active = BooleanProperty(False)

    # Internal state
    _texture = ObjectProperty(None, allownone=True)
    _surface = ObjectProperty(None, allownone=True)
    _buffer = ObjectProperty(None, allownone=True)
    _connection = ObjectProperty(None, allownone=True)

    def __init__(self, particle_system=None, **kwargs):
        super().__init__(**kwargs)

        self.particle_system = particle_system
        self._render_thread = None
        self._stop_rendering = threading.Event()
        self._memory_pointer = None
        self._render_callback = None

        # Initialize graphics
        with self.canvas:
            self._texture = Texture.create(
                size=(int(self.surface_width), int(self.surface_height)),
                colorfmt="rgba",
            )
            self._rect = Rectangle(texture=self._texture, pos=self.pos, size=self.size)

        self.bind(pos=self._update_rect, size=self._update_rect)
        self.bind(surface_width=self._on_surface_size_change)
        self.bind(surface_height=self._on_surface_size_change)

        if SDL2_AVAILABLE:
            Clock.schedule_once(self._init_native_surface, 0)

    def _update_rect(self, *args):
        """Update the rectangle position and size"""
        if hasattr(self, "_rect"):
            self._rect.pos = self.pos
            self._rect.size = self.size

    def _on_surface_size_change(self, *args):
        """Handle surface size changes"""
        if self.is_active:
            self.stop_rendering()
            Clock.schedule_once(lambda dt: self.start_rendering(), 0.1)

    def _init_native_surface(self, dt):
        """Initialize the native SDL2 surface and termuxgui buffer"""
        if not SDL2_AVAILABLE:
            Logger.warning("SDL2 not available, using fallback rendering")
            return

        try:
            # Initialize termuxgui connection
            self._connection = tg.Connection()

            # Create buffer for the surface
            width, height = int(self.surface_width), int(self.surface_height)
            self._buffer = tg.Buffer(self._connection, width, height)

            # Initialize SDL2
            sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

            # Create SDL surface from buffer
            self._create_sdl_surface()

            Logger.info(f"Native surface initialized: {width}x{height}")

        except Exception as e:
            Logger.error(f"Failed to initialize native surface: {e}")
            SDL2_AVAILABLE = False

    def _create_sdl_surface(self):
        """Create SDL2 surface from termuxgui buffer"""
        if not self._buffer:
            return

        width, height = int(self.surface_width), int(self.surface_height)

        # Get memory pointer from buffer
        self._buffer_context = self._buffer.__enter__()
        self._memory_pointer = ctypes.cast(
            ctypes.pointer(ctypes.c_uint8.from_buffer(self._buffer_context, 0)),
            ctypes.c_void_p,
        )

        # Create SDL surface
        self._surface = sdl2.SDL_CreateRGBSurfaceFrom(
            self._memory_pointer,
            width,
            height,
            32,  # 32-bit depth (ARGB8888)
            4 * width,  # pitch (bytes per row)
            ctypes.c_uint(0xFF),  # Red mask
            ctypes.c_uint(0xFF00),  # Green mask
            ctypes.c_uint(0xFF0000),  # Blue mask
            ctypes.c_uint(0xFF000000),  # Alpha mask
        )

    def set_render_callback(self, callback: Callable[[object], None]):
        """Set custom render callback function"""
        self._render_callback = callback

    def start_rendering(self):
        """Start the native rendering thread"""
        if not SDL2_AVAILABLE or self.is_active:
            return

        self._stop_rendering.clear()
        self.is_active = True

        if SDL2_AVAILABLE:
            self._render_thread = threading.Thread(
                target=self._render_loop, daemon=True
            )
            self._render_thread.start()
        else:
            # Fallback to Kivy rendering
            self._fallback_rendering()

    def stop_rendering(self):
        """Stop the native rendering thread"""
        if not self.is_active:
            return

        self.is_active = False
        self._stop_rendering.set()

        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=0.5)

    def _render_loop(self):
        """Main rendering loop running in separate thread"""
        if not self._surface or not self._buffer:
            Logger.error("Surface not properly initialized")
            return

        frame_time = 1.0 / self.fps_target

        while not self._stop_rendering.is_set() and self.is_active:
            start_time = time.time()

            try:
                # Clear surface
                self._clear_surface()

                # Render particles
                if self._render_callback:
                    self._render_callback(self._surface)
                elif self.particle_system:
                    self._render_particles()
                else:
                    self._render_test_pattern()

                # Update Kivy texture
                Clock.schedule_once(self._update_kivy_texture, 0)

                # Frame rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                Logger.error(f"Rendering error: {e}")
                break

    def _clear_surface(self):
        """Clear the SDL surface"""
        if self._surface:
            black = sdl2.ext.Color(0, 0, 0, 255)
            sdl2.ext.fill(self._surface, black)

    def _render_particles(self):
        """Render particles using SDL2"""
        if not self.particle_system or not self._surface:
            return

        # Get particles from system
        particles = getattr(self.particle_system, "particles", [])

        for particle in particles:
            # Extract particle properties
            x = int(getattr(particle, "x", 0))
            y = int(getattr(particle, "y", 0))
            size = int(getattr(particle, "size", 2))

            # Get particle color (default to white)
            if hasattr(particle, "color"):
                r, g, b, a = particle.color
                color = sdl2.ext.Color(
                    int(r * 255), int(g * 255), int(b * 255), int(a * 255)
                )
            else:
                color = sdl2.ext.Color(255, 255, 255, 255)

            # Render particle as rectangle
            rect = (x - size // 2, y - size // 2, size, size)
            sdl2.ext.fill(self._surface, color, rect)

    def _render_test_pattern(self):
        """Render a test pattern for debugging"""
        if not self._surface:
            return

        width, height = int(self.surface_width), int(self.surface_height)

        # Create some moving squares
        t = time.time()

        red = sdl2.ext.Color(255, 0, 0, 255)
        green = sdl2.ext.Color(0, 255, 0, 255)
        blue = sdl2.ext.Color(0, 0, 255, 255)

        # Moving red square
        x1 = int((width - 20) * (0.5 + 0.5 * time.sin(t)))
        sdl2.ext.fill(self._surface, red, (x1, height // 4, 20, 20))

        # Moving green square
        y1 = int((height - 20) * (0.5 + 0.5 * time.sin(t * 1.5)))
        sdl2.ext.fill(self._surface, green, (width // 2, y1, 20, 20))

        # Moving blue square
        x2 = int((width - 20) * (0.5 + 0.5 * time.sin(t * 0.7)))
        y2 = int((height - 20) * (0.5 + 0.5 * time.cos(t * 0.7)))
        sdl2.ext.fill(self._surface, blue, (x2, y2, 20, 20))

    def _update_kivy_texture(self, dt):
        """Update the Kivy texture with SDL surface data"""
        if not self._surface or not self._buffer:
            return

        try:
            # Copy SDL surface data to Kivy texture
            width, height = int(self.surface_width), int(self.surface_height)

            # Get pixel data from SDL surface
            pixels = ctypes.cast(
                self._surface.contents.pixels, ctypes.POINTER(ctypes.c_uint8)
            )

            # Convert to bytes
            pixel_data = ctypes.string_at(pixels, width * height * 4)

            # Update Kivy texture
            if self._texture:
                self._texture.blit_buffer(
                    pixel_data, colorfmt="rgba", bufferfmt="ubyte"
                )

        except Exception as e:
            Logger.error(f"Texture update error: {e}")

    def _fallback_rendering(self):
        """Fallback to Kivy-based rendering when SDL2 is not available"""

        def update_fallback(dt):
            if not self.is_active:
                return False

            # Simple fallback rendering using Kivy
            if self._texture:
                width, height = int(self.surface_width), int(self.surface_height)
                # Create simple test pattern
                import random

                pixels = bytearray(width * height * 4)

                for i in range(0, len(pixels), 4):
                    pixels[i] = random.randint(0, 255)  # R
                    pixels[i + 1] = random.randint(0, 255)  # G
                    pixels[i + 2] = random.randint(0, 255)  # B
                    pixels[i + 3] = 255  # A

                self._texture.blit_buffer(
                    bytes(pixels), colorfmt="rgba", bufferfmt="ubyte"
                )

            return True

        Clock.schedule_interval(update_fallback, 1.0 / self.fps_target)

    def cleanup(self):
        """Clean up resources"""
        self.stop_rendering()

        # Unbind window events
        if self.auto_resize:
            try:
                Window.unbind(size=self._on_window_resize)
            except:
                pass  # Already unbound or window destroyed

        self._cleanup_native_resources()

        # Force garbage collection
        gc.collect()

    def _cleanup_native_resources(self):
        """Clean up native SDL/buffer resources"""
        # Clean up SDL resources
        if self._surface:
            sdl2.SDL_FreeSurface(self._surface)
            self._surface = None

        if self._memory_pointer:
            del self._memory_pointer
            self._memory_pointer = None

        if hasattr(self, "_buffer_context") and self._buffer_context:
            try:
                self._buffer.__exit__(None, None, None)
            except:
                pass  # May already be closed

        if self._buffer:
            self._buffer = None

        if self._connection:
            try:
                self._connection.close()
            except:
                pass  # May already be closed
            self._connection = None

        if SDL2_AVAILABLE:
            try:
                sdl2.SDL_Quit()
            except:
                pass  # SDL may already be quit

    def __del__(self):
        """Destructor"""
        self.cleanup()
