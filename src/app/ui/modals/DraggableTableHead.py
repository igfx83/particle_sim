from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.properties import (
    NumericProperty,
    StringProperty,
    BooleanProperty,
)
from kivy.animation import Animation


class DraggableTableHead(Label):
    """Custom table header that can be dragged to reorder columns"""

    column = StringProperty("")
    original_index = NumericProperty(0)
    is_dragging = BooleanProperty(False)
    drag_offset_x = NumericProperty(0)
    drag_offset_y = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_pos = (0, 0)
        self.drag_start_pos = (0, 0)
        if self.canvas:
            with self.canvas.before:
                Color((0.5, 0.5, 0.5, 1))  # Gray background
                Rectangle(pos=self.pos, size=self.size)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Start dragging
            self.is_dragging = True
            self.original_pos = self.pos
            self.drag_start_pos = touch.pos
            self.drag_offset_x = touch.pos[0] - self.center_x
            self.drag_offset_y = touch.pos[1] - self.center_y

            # Bring to front
            parent = self.parent
            if parent:
                parent.remove_widget(self)
                parent.add_widget(self)

            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self and self.is_dragging:
            # Update position during drag
            self.center_x = touch.pos[0] - self.drag_offset_x
            self.center_y = touch.pos[1] - self.drag_offset_y

            # Notify parent about drag position for column reordering preview
            if hasattr(self.parent, "on_header_drag"):
                self.parent.on_header_drag(self, touch.pos)

            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self and self.is_dragging:
            self.is_dragging = False

            # Find drop target
            drop_target = None
            if hasattr(self.parent, "get_drop_target"):
                drop_target = self.parent.get_drop_target(self, touch.pos)

            if drop_target and drop_target != self:
                # Perform column reorder
                if hasattr(self.parent, "reorder_columns"):
                    self.parent.reorder_columns(self.column, drop_target.column)
            else:
                # Snap back to original position
                Animation(pos=self.original_pos, duration=0.2).start(self)

            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)
