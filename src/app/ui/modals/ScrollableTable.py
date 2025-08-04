# pyright: reportAttributeAccessIssue=false
from app.ui.elements import load_elements
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from app.ui.modals.DraggableTableHead import DraggableTableHead

ELEMENTS = load_elements()


class ScrollableTable(ScrollView):
    """Custom scrollable table with proper drag handling"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_scroll_x = True
        self.do_scroll_y = True
        self.bar_width = 10
        self.scroll_type = ["bars", "content"]
        self.bind(size=self._update_scroll_y)

        # Main container for headers and data
        self.main_layout = BoxLayout(orientation="vertical", size_hint=(None, None))
        self.main_layout.bind(minimum_width=self.main_layout.setter("width"))
        self.main_layout.bind(minimum_height=self.main_layout.setter("height"))

        # Header container (non-scrollable in Y)
        self.header_container = Widget(size_hint=(None, None), height=40)
        self.headers = []

        # Data grid (scrollable)
        self.data_grid = GridLayout(cols=1, size_hint=(None, None), spacing=(1, 1))
        self.data_grid.bind(minimum_width=self.data_grid.setter("width"))
        self.data_grid.bind(minimum_height=self.data_grid.setter("height"))

        # Add header and data to main layout
        self.main_layout.add_widget(self.header_container)
        self.main_layout.add_widget(self.data_grid)

        # Add main layout to ScrollView
        self.add_widget(self.main_layout)

    def _update_scroll_y(self, instance, value):
        # Only enable vertical scrolling if content height exceeds ScrollView height
        content_height = self.main_layout.height
        if content_height <= self.height:
            self.do_scroll_y = False
        else:
            self.do_scroll_y = True

    def clear_table(self):
        """Clear all table content"""
        self.header_container.clear_widgets()
        self.data_grid.clear_widgets()
        self.headers.clear()

    def set_columns(self, num_cols):
        """Set the number of columns"""
        self.data_grid.cols = num_cols
        self._update_header_layout()

    def add_header(self, text, column_id):
        header = DraggableTableHead(
            text=text,
            column=column_id,
            size_hint=(None, None),
            height=40,
        )
        header.original_index = len(self.headers)
        self.headers.append(header)
        self.header_container.add_widget(header)
        self._update_header_layout()
        return header

    def add_data_widget(self, widget):
        """Add a data widget to the table"""
        self.data_grid.add_widget(widget)

    def _update_header_layout(self):
        if not self.headers:
            return

        num_cols = len(self.headers)
        if num_cols == 0:
            return

        # Calculate column width
        total_width = max(self.width, num_cols * 100)  # Minimum 100px per column
        col_width = total_width / num_cols

        # Position headers
        for i, header in enumerate(self.headers):
            header.width = col_width
            header.x = i * col_width
            header.y = 0  # Headers stay at top

        # Update container sizes
        self.header_container.width = total_width
        self.header_container.height = 40
        self.data_grid.width = total_width
        self.main_layout.width = total_width
        self.main_layout.height = self.header_container.height + self.data_grid.height

    def on_header_drag(self, header, pos):
        """Handle header drag for visual feedback"""
        # Could add visual indicators here
        pass

    def get_drop_target(self, dragged_header, pos):
        """Find which header is being dragged over"""
        for header in self.headers:
            if header != dragged_header and header.collide_point(*pos):
                return header
        return None

    def reorder_columns(self, from_column, to_column):
        """Reorder columns and notify parent"""
        if hasattr(self.parent, "reorder_columns"):
            self.parent.reorder_columns(from_column, to_column)

    def on_touch_down(self, touch):
        # Check if touch is on a header first
        for header in self.headers:
            if header.collide_point(*touch.pos):
                return header.on_touch_down(touch)

        # Otherwise, handle normal scrolling
        return super().on_touch_down(touch)

    def on_scroll_stop(self, touch, check_children=True):
        # Clamp scroll position to prevent overscroll
        self.scroll_x = max(0, min(self.scroll_x, 1))
        self.scroll_y = max(0, min(self.scroll_y, 1))
        return super().on_scroll_stop(touch, check_children)
