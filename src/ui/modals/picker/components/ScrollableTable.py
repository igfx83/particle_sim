# pyright: reportAttributeAccessIssue=false
# from kivy.app import App
# from kivy.logger import Logger
# from kivy.uix.label import NumericProperty
# from ui.widgets import PickerLabel
# from kivy.properties import ObjectProperty
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.scrollview import ScrollView
# from kivy.clock import Clock
# from ui.modals.picker.components.DraggableTableHead import DraggableTableHead
#
# pyright: reportAttributeAccessIssue=false
import weakref
from typing import Dict, List, Any
from kivy.app import App
from kivy.logger import Logger
from ui.widgets import PickerLabel
from kivy.properties import (
    ObjectProperty,
    ListProperty,
    NumericProperty,
    BooleanProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from ui.modals.picker.components.DraggableTableHead import DraggableTableHead


class ColumnBoxLayout(BoxLayout):
    """Optimized BoxLayout for table columns with caching"""

    max_width = NumericProperty(100)
    _cached_children = ListProperty([])
    _width_dirty = BooleanProperty(True)

    def __init__(self, id: str, **kwargs):
        super(ColumnBoxLayout, self).__init__(**kwargs)

        self.bind(minimum_height=self.setter("height"))
        self.index = 0
        self.id = id
        self._last_texture_sizes = {}

    def add_widget(self, widget, **kwargs):
        """Optimized widget addition with batched updates"""
        if isinstance(widget, (PickerLabel, DraggableTableHead)):
            widget.size_hint_x = None
            widget.index = len(self.children)
            # Cache texture size on creation to avoid repeated calculations
            if hasattr(widget, "texture_size"):
                self._last_texture_sizes[id(widget)] = widget.texture_size

        super().add_widget(widget, **kwargs)
        self._cached_children.append(weakref.ref(widget))
        self._width_dirty = True

    def get_optimal_width(self) -> float:
        """Vectorized width calculation with caching"""
        if not self._width_dirty:
            return self.max_width

        widths = []
        for widget_ref in self._cached_children:
            widget = widget_ref()
            if widget and hasattr(widget, "texture_size"):
                # Use cached texture size if available and text hasn't changed
                widget_id = id(widget)
                if (
                    widget_id in self._last_texture_sizes
                    and hasattr(widget, "_last_text")
                    and widget._last_text == getattr(widget, "text", "")
                ):
                    texture_width = self._last_texture_sizes[widget_id][0]
                else:
                    texture_width = widget.texture_size[0]
                    self._last_texture_sizes[widget_id] = widget.texture_size
                    if hasattr(widget, "text"):
                        widget._last_text = widget.text

                widths.append(texture_width + 10)  # 10px padding

        self.max_width = max(widths)  # Minimum 100px
        self._width_dirty = False
        return self.max_width


class ScrollableTable(ScrollView):
    """Optimized scrollable table with vectorized operations"""

    _cols = ObjectProperty({})
    _property_cache = ObjectProperty(None, allownone=True)
    _columns_cache = ListProperty([])

    def __init__(self, **kwargs):
        super(ScrollableTable, self).__init__(**kwargs)

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

        # Defer heavy operations
        Clock.schedule_once(self._initialize_table, 0)

    # TODO:Debug and restore scrollX functionality
    def _initialize_table(self, dt):
        """Deferred table initialization for better startup performance"""
        properties = self._get_cached_properties()
        if properties:
            self._create_columns_batch(properties)

    def _get_cached_properties(self) -> Dict[str, List[Any]]:
        """Cached property extraction with vectorized processing"""
        if self._property_cache is not None:
            return self._property_cache

        self.app = App.get_running_app()
        if not self.app or not hasattr(self.app, "elements"):
            Logger.error("No running app or elements found")
            return {}

        # Vectorized property extraction
        properties = {}
        target_keys = {
            "radius",
            "ignition_temperature",
            "phase_transitions",
            "flame_temperature",
            "burn_duration",
            "combustion_products",
            "oxygen_requirement",
        }

        # Process all elements in one pass
        for element_data in self.app.elements.values():
            intrinsic = element_data.get("intrinsic_properties", {})

            for key, value in intrinsic.items():
                if key in target_keys:
                    if key == "phase_transitions":
                        self._process_phase_transitions(value, properties)
                    continue
                properties.setdefault(key, []).append(value)

        self._property_cache = properties
        return properties

    def _process_phase_transitions(self, transitions: Dict, properties: Dict):
        """Optimized phase transition processing"""
        if all(phase in transitions for phase in ["freezing", "boiling"]):
            for transition in ["boiling", "freezing"]:
                if transition in transitions:
                    temp = transitions[transition][0]["temperature"]
                    properties.setdefault(transition, []).append(temp)
        else:
            properties.setdefault("boiling", []).append("N/A")
            properties.setdefault("freezing", []).append("N/A")

    def _create_columns_batch(self, properties: Dict[str, List[Any]]):
        """Vectorized column creation with batched layout updates"""
        columns = []

        # Create all columns first (batch creation)
        for i, (head_text, data) in enumerate(properties.items()):
            column = self._create_column_optimized(head_text, data, i)
            columns.append(column)

        # Batch add to container (single layout pass)
        for column in columns:
            self.container.add_widget(column)

        # Single deferred width calculation for all columns
        Clock.schedule_once(lambda dt: self._batch_update_widths(columns), 0.1)

    def _create_column_optimized(
        self, head_text: str, data: List[Any], index: int
    ) -> ColumnBoxLayout:
        """Optimized column creation with minimal layout updates"""
        column = ColumnBoxLayout(
            id=head_text, orientation="vertical", size_hint=(None, None)
        )

        # Create header
        header = DraggableTableHead(
            column, text=head_text, size_hint=(None, None), height=40
        )
        column.add_widget(header)
        self._cols[header] = data

        # Batch create labels
        labels = self._create_labels_batch(data, head_text)

        # Batch add labels (single layout update)
        for label in labels:
            column.add_widget(label)

        column.index = index
        return column

    def _create_labels_batch(
        self, data: List[Any], head_text: str
    ) -> List[PickerLabel]:
        """Vectorized label creation"""
        labels = []

        if head_text == "color":
            # Batch create color labels
            labels = [
                PickerLabel(
                    text="",
                    color=(0, 0, 0, 1),
                    background_color=color,
                    size_hint=(None, None),
                    height=40,
                )
                for color in data
            ]
        else:
            # Batch create text labels
            labels = [
                PickerLabel(text=str(item), size_hint=(None, None), height=40)
                for item in data
            ]

        return labels

    def _batch_update_widths(self, columns: List[ColumnBoxLayout]):
        """Vectorized width updates for all columns"""
        # Calculate optimal widths for all columns at once
        optimal_widths = [col.get_optimal_width() for col in columns]

        # Apply widths in batch
        for column, width in zip(columns, optimal_widths):
            self._apply_column_width(column, width)

        # Single layout update for container
        self.container.do_layout()

    def _apply_column_width(self, column: ColumnBoxLayout, width: float):
        """Apply width to column and all its children efficiently"""
        column.width = width
        column.max_width = width

        # Batch update all children widths
        for child in column.children:
            if hasattr(child, "width"):
                child.width = width

    def _update_scroll_y(self, instance, value):
        """Optimized scroll update check"""
        self.do_scroll_y = self.container.height > self.height

    # def on_touch_down(self, touch):
    #     """Optimized touch handling with early exit"""
    #     # Quick bounds check first
    #     if not self.collide_point(*touch.pos):
    #         return False
    #
    #     # Check headers only if within table bounds
    #     for header in self._cols.keys():
    #         if header.collide_point(*touch.pos):
    #             return header.on_touch_down(touch)
    #
    #     return super().on_touch_down(touch)

    def _sort_columns(self, column):
        """Optimized sorting with minimal widget manipulation"""
        if not hasattr(self, "_sort_reverse"):
            self._sort_reverse = False
        self._sort_reverse = not self._sort_reverse

        # Get non-header labels
        column_labels = [
            label
            for label in column.children
            if not isinstance(label, DraggableTableHead)
        ]

        # Optimized sort key function
        def sort_key(label):
            text = getattr(label, "text", "")
            try:
                return (0, float(text))
            except (ValueError, TypeError):
                return (1, text)

        # Create sort indices (avoid moving widgets during sort)
        indexed_labels = list(enumerate(column_labels))
        indexed_labels.sort(key=lambda x: sort_key(x[1]), reverse=self._sort_reverse)
        new_order = [index for index, _ in indexed_labels]

        # Batch update all columns using the same sort order
        self._apply_sort_to_all_columns(new_order)

    def _apply_sort_to_all_columns(self, new_order: List[int]):
        """Apply sort order to all columns efficiently"""
        for column in self.container.children:
            header = next(
                (w for w in column.children if isinstance(w, DraggableTableHead)), None
            )
            if not header:
                continue

            labels = [
                label
                for label in column.children
                if not isinstance(label, DraggableTableHead)
            ]

            # Minimize widget operations
            column.clear_widgets()
            column.add_widget(header)

            # Add labels in new order
            for i in new_order:
                if i < len(labels):
                    column.add_widget(labels[i])

    def invalidate_cache(self):
        """Force cache invalidation for dynamic updates"""
        self._property_cache = None
        for column in self.container.children:
            column._width_dirty = True
