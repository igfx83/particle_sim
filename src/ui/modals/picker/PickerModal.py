# pyright: reportAttributeAccessIssue=false
# from kivy.logger import Logger
from kivy.app import App
from kivy.cache import Cache
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from ui.elements import load_elements
from ui.modals.picker.components.ScrollableTable import ScrollableTable
from kivy.uix.modalview import ModalView

ELEMENTS = load_elements()


class PickerModal(ModalView):
    element_table = ObjectProperty({})

    def __init__(self, **kwargs):
        super(PickerModal, self).__init__(**kwargs)
        self.app: App = App.get_running_app()
        self._table_initialized = False
        self._pending_operations = []

    def _get_or_create_table(self) -> ScrollableTable:
        """Efficient table creation with proper caching"""
        # Try to get from cache first
        cached_table = Cache.get("resources", "element_table")

        if cached_table and hasattr(cached_table, "container"):
            # Validate cached table is still functional
            if cached_table.parent is None:  # Not currently attached
                return cached_table

        # Create new table if cache miss or invalid
        new_table = ScrollableTable()
        self._table_initialized = True
        Cache.append("resources", "element_table", new_table, timeout=0)  # 5min timeout
        return new_table

    def on_touch_move(self, touch):
        """Optimized touch handling - removed debug logging for performance"""
        return super().on_touch_move(touch)

    def on_pre_open(self):
        """Optimized pre-open with deferred table loading"""
        if not self.app or not hasattr(self.app, "root"):
            return

        # Defer heavy table operations
        Clock.schedule_once(self._load_table_deferred, 0)

    def _load_table_deferred(self, dt):
        """Deferred table loading to prevent UI blocking"""
        if not self.element_table:
            self.element_table = self._get_or_create_table()

        # Add table to modal
        panel = self.ids.get("picker_modal_panel")
        if panel and self.element_table:
            # Ensure table isn't already added
            if self.element_table.parent != panel:
                # Remove from previous parent if any
                if self.element_table.parent:
                    self.element_table.parent.remove_widget(self.element_table)
                panel.add_widget(self.element_table)

    def on_pre_dismiss(self):
        """Optimized cleanup with proper caching"""
        if not self.app or not self.app.root.ids.simulation_cursor:
            return

        # Immediate UI state restoration
        self.app.is_running = True
        self.app.root.ids.simulation_cursor._modal_open = False

        # Cache the table for reuse
        if self.element_table:
            # Remove from current parent before caching
            if self.element_table.parent:
                self.element_table.parent.remove_widget(self.element_table)

            # Update cache with current table state
            Cache.append("resources", "element_table", self.element_table, timeout=300)

            # Clean up any cached references in the table
            if hasattr(self.element_table, "invalidate_cache"):
                Clock.schedule_once(
                    lambda dt: self.element_table.invalidate_cache(), 0.1
                )

    def refresh_table_data(self):
        """Force refresh table data (call when elements change)"""
        # Clear cache to force recreation
        Cache.remove("resources", "element_table")

        if self.element_table:
            # Clean up current table
            if hasattr(self.element_table, "invalidate_cache"):
                self.element_table.invalidate_cache()

            self.element_table = None
            self._table_initialized = False

        # If modal is open, recreate table immediately
        if hasattr(self, "_in_open_state") and self._in_open_state:
            Clock.schedule_once(self._load_table_deferred, 0)

    def on_open(self):
        """Track open state for refresh operations"""
        super().on_open()
        self._in_open_state = True

    def on_dismiss(self):
        """Track open state for refresh operations"""
        super().on_dismiss()
        self._in_open_state = False
