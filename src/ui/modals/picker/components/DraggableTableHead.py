# pyright: reportOperatorIssue=false
import time
from kivy.uix.label import Label
from kivy.input.motionevent import MotionEvent
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty
from kivy.clock import Clock
# from kivy.animation import Animation


class DraggableTableHead(Label):
    """Custom table header that can be dragged to reorder columns"""

    column = ObjectProperty(None)
    is_dragging = BooleanProperty(False)
    drag_threshold = NumericProperty(5)  # Minimum distance to start drag
    tap_timeout = NumericProperty(0.2)  # Maximum time for tap detection

    def __init__(self, column, **kwargs):
        super().__init__(**kwargs)
        self.column = column
        self.column.id = self.text
        self.bind(pos=self._schedule_rect_update, size=self._schedule_rect_update)

        # Optimization: Cache frequently accessed values
        self._parent_ref = None
        self._siblings_cache = []
        self._rect_update_scheduled = False

    def _schedule_rect_update(self, *args):
        """Debounced rect updates to avoid excessive calls"""
        if not self._rect_update_scheduled:
            self._rect_update_scheduled = True
            Clock.schedule_once(self._update_rect_deferred, 0)

    def _update_rect_deferred(self, dt):
        """Deferred rect update"""
        self._rect_update_scheduled = False
        if hasattr(self, "rect"):
            self.rect.pos = self.pos
            self.rect.size = self.size

    def on_touch_down(self, touch: MotionEvent):
        """Optimized touch down with early collision detection"""
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)

        # Cache parent and siblings for drag operations
        self._parent_ref = self.column.parent
        if self._parent_ref:
            self._siblings_cache = list(self._parent_ref.children)

        # Initialize touch data efficiently
        touch.ud.update(
            {
                "neighbor": None,
                "distance_traveled": 0,
                "columns": self._siblings_cache,
                "start_time": time.time(),
                "moved": False,
            }
        )

        self.is_dragging = True
        touch.grab(self)
        return True

    def on_touch_move(self, touch: MotionEvent):
        """Optimized touch move with reduced calculations"""
        if touch.grab_current is not self or not self.is_dragging:
            return super().on_touch_move(touch)

        parent = self._parent_ref
        if not parent:
            return True

        # Mark as moved for tap detection
        if abs(touch.dx) > 2 or abs(touch.dy) > 2:
            touch.ud["moved"] = True

        siblings = touch.ud["columns"]
        if not siblings or self.column not in siblings:
            return True

        current_index = siblings.index(self.column)
        touch.ud["distance_traveled"] += touch.dx

        # Determine neighbor only when threshold is crossed
        if (
            touch.ud["neighbor"] is None
            and abs(touch.ud["distance_traveled"]) > self.drag_threshold
        ):
            if touch.ud["distance_traveled"] > 0 and current_index > 0:
                touch.ud["neighbor"] = siblings[current_index - 1]
            elif (
                touch.ud["distance_traveled"] < 0 and current_index < len(siblings) - 1
            ):
                touch.ud["neighbor"] = siblings[current_index + 1]

        neighbor = touch.ud["neighbor"]
        if neighbor and self._should_swap_columns(touch, neighbor, current_index):
            self._perform_column_swap(parent, current_index, touch)
            return True

        # Smooth column dragging
        if neighbor:
            self._update_column_positions(touch, neighbor)

        return True

    def _should_swap_columns(
        self, touch: MotionEvent, neighbor, current_index: int
    ) -> bool:
        """Optimized collision detection for column swapping"""
        if self.column.collide_widget(neighbor):
            return False

        distance_threshold = neighbor.width
        horizontal_movement = abs(touch.x - touch.ox)

        return horizontal_movement >= distance_threshold

    def _perform_column_swap(self, parent, current_index: int, touch: MotionEvent):
        """Efficient column reordering"""
        direction = 1 if touch.x < touch.ox else -1
        new_index = max(
            0, min(len(self._siblings_cache) - 1, current_index + direction)
        )

        # Batch widget operations
        parent.remove_widget(self.column)
        parent.add_widget(self.column, index=new_index)

        # Reset drag state
        touch.ud["neighbor"] = None
        touch.ud["distance_traveled"] = 0

        # Update cached siblings list
        self._siblings_cache = list(parent.children)

    def _update_column_positions(self, touch: MotionEvent, neighbor):
        """Smooth position updates during drag"""
        if not neighbor:
            return

        # Calculate position adjustments
        width_ratio = self.column.width / neighbor.width if neighbor.width > 0 else 1
        adjustment = touch.dx * width_ratio

        # Apply position changes
        self.column.x += touch.dx
        neighbor.x -= adjustment

    def on_touch_up(self, touch: MotionEvent):
        """Optimized touch up with tap detection"""
        if touch.grab_current is not self or not self.is_dragging:
            return super().on_touch_up(touch)

        # Detect tap vs drag
        duration = time.time() - touch.ud.get("start_time", 0)
        is_tap = (
            duration < self.tap_timeout
            and not touch.ud.get("moved", False)
            and abs(touch.x - touch.ox) < self.drag_threshold
        )

        if is_tap:
            self._handle_tap()
        else:
            self._handle_drag_end()

        # Cleanup
        self.is_dragging = False
        touch.ungrab(self)
        return True

    def _handle_tap(self):
        """Handle tap events (sorting)"""
        table = self.column.parent.parent if self.column.parent else None
        if table and hasattr(table, "_sort_columns"):
            # Use Clock to avoid blocking the UI
            Clock.schedule_once(lambda dt: table._sort_columns(self.column), 0)

    def _handle_drag_end(self):
        """Handle drag end cleanup"""
        # Snap header position to column position
        if self.column:
            self.x = self.column.x

    def cleanup_cache(self):
        """Clean up cached references to prevent memory leaks"""
        self._parent_ref = None
        self._siblings_cache.clear()
