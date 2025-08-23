# pyright: reportAttributeAccessIssue=false
from kivy.uix.label import NumericProperty
from app.ui.elements import load_elements
from app.ui.widgets import PickerLabel
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from app.ui.modals.DraggableTableHead import DraggableTableHead

ELEMENTS = load_elements()


class ColumnBoxLayout(BoxLayout):
    """Custom BoxLayout for table columns with a max_width property"""

    max_width = NumericProperty(100)  # Default width

    def __init__(self, id: str, **kwargs):
        super().__init__(**kwargs)
        self.bind(minimum_height=self.setter("height"))
        self.index = 0
        self.id = id

    def add_widget(self, widget, **kwargs):
        """Override to ensure all children have size_hint_x set to None"""
        if isinstance(widget, (PickerLabel, DraggableTableHead)):
            widget.size_hint_x = None
            widget.index = len(self.children)
        super().add_widget(widget, **kwargs)


class ScrollableTable(ScrollView):
    """Custom scrollable table with proper drag handling"""

    _cols = ObjectProperty({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.container = BoxLayout(
            orientation="horizontal", size_hint=(None, None), spacing=0
        )
        self.container.bind(
            minimum_width=self.container.setter("width"),
            minimum_height=self.container.setter("height"),
        )
        super().add_widget(self.container)
        self.do_scroll_x, self.do_scroll_y = True, True
        self.bar_width = 10
        self.scroll_type = ["bars", "content"]
        self.bind(size=self._update_scroll_y)

        properties = self._get_property_dict(ELEMENTS)
        for head_text, data in properties.items():
            column = self._create_column(head_text, data)
            column.index = len(self.container.children)
            self.container.add_widget(column)

    def _get_property_dict(self, elements) -> dict:
        properties = {}
        for _, element in elements.items():
            for k, v in element["intrinsic_properties"].items():
                if k in [
                    "radius",
                    "phase_transitions",
                    "ignition_temperature",
                    "flame_temperature",
                    "burn_duration",
                    "combustion_products",
                    "oxygen_requirement",
                ]:
                    continue
                properties.setdefault(k, []).append(v)
        return properties

    def _create_column(self, head_text, data) -> ColumnBoxLayout:
        column = ColumnBoxLayout(
            id=head_text, orientation="vertical", size_hint=(None, None)
        )
        header = DraggableTableHead(
            column, text=head_text, size_hint=(None, None), height=40
        )
        column.add_widget(header)
        self._cols[header] = data
        labels = []

        if head_text == "color":
            for color in data:
                label = PickerLabel(
                    text="",
                    color=(0, 0, 0, 1),
                    background_color=color,
                    size_hint=(None, None),
                    height=40,
                )
                column.add_widget(label)
                labels.append(label)
        else:
            for item in data:
                data_label = PickerLabel(
                    text=str(item), size_hint=(None, None), height=40
                )
                column.add_widget(data_label)
                labels.append(data_label)

        def update_widths(dt):
            max_width = max(header.texture_size[0] + 10, 100)
            for label in labels:
                max_width = max(max_width, label.texture_size[0] + 10)
            header.width = max_width
            for label in labels:
                label.width = max_width
                column.max_width = column.width = max_width
                column.do_layout()

        self.container.do_layout()

        Clock.schedule_once(update_widths, 0)
        return column

    def _update_scroll_y(self, instance, value):
        self.do_scroll_y = self.container.height > self.height

    def on_touch_down(self, touch):
        for header in self._cols.keys():
            if header.collide_point(*touch.pos):
                return header.on_touch_down(touch)
        return super().on_touch_down(touch)
