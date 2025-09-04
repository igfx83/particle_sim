# pyright: reportOperatorIssue=false
import time
from kivy.uix.label import Label
from kivy.input.motionevent import MotionEvent
from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
)

# from kivy.animation import Animation


class DraggableTableHead(Label):
    """Custom table header that can be dragged to reorder columns"""

    column = ObjectProperty(None)
    is_dragging = BooleanProperty(False)

    def __init__(self, column, **kwargs):
        super().__init__(**kwargs)
        self.column = column
        self.column.id = self.text
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        if hasattr(self, "rect"):
            self.rect.pos = self.pos
            self.rect.size = self.size

    def on_touch_down(self, touch: MotionEvent):
        if self.collide_point(*touch.pos):
            self.is_dragging = True
            touch.ud["neighbor"] = None
            touch.ud["distance_traveled"] = 0
            touch.ud["columns"] = list(self.column.parent.children)
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch: MotionEvent):
        if touch.grab_current is self and self.is_dragging:
            parent = self.column.parent
            if not parent:
                return True

            siblings = touch.ud["columns"]

            index = siblings.index(self.column)

            touch.ud["distance_traveled"] += touch.dx

            if touch.ud["neighbor"] is None:
                if touch.ud["distance_traveled"] > 0:
                    touch.ud["neighbor"] = (
                        self.column.parent.children[index - 1] if index > 0 else None
                    )
                    touch.ud["limit"] = self.column.right
                elif touch.ud["distance_traveled"] < 0:
                    touch.ud["neighbor"] = (
                        self.column.parent.children[index + 1]
                        if index < len(self.column.parent.children) - 1
                        else None
                    )
                    touch.ud["limit"] = self.column.x

            if touch.ud["neighbor"]:
                if self.column.collide_widget(touch.ud["neighbor"]) is False:
                    if (
                        touch.x < touch.ox
                        and touch.x < touch.px
                        and abs(touch.x - touch.ox) >= touch.ud["neighbor"].width
                    ):
                        parent.remove_widget(self.column)
                        parent.add_widget(self.column, index=index + 1)
                    elif (
                        touch.x > touch.ox
                        and touch.x > touch.px
                        and abs(touch.x - touch.ox) >= touch.ud["neighbor"].width
                    ):
                        parent.remove_widget(self.column)
                        parent.add_widget(self.column, index=index - 1)
                    touch.ud["neighbor"] = None
                    touch.ud["distance_traveled"] = 0
                    return True

                self.column.x += touch.dx
                touch.ud["neighbor"].x -= touch.dx * (
                    self.column.width / touch.ud["neighbor"].width
                )
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch: MotionEvent):
        if touch.grab_current is self and self.is_dragging:
            current_time = time.time()
            duration = current_time - touch.time_start
            if duration < 0.2 and abs(touch.x - touch.ox) < 5:
                # Considered a tap, not a drag
                self.column.parent.parent._sort_columns(self.column)
            self.is_dragging = False
            self.x = self.column.x
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)
