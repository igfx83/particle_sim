from kivy.event import EventDispatcher
from kivy.storage.jsonstore import JsonStore as Store
from kivy.properties import ListProperty


class AppState(EventDispatcher):
    """Singleton class to manage application state."""

    column_order = ListProperty([])

    def __init__(self):
        super().__init__()
        self._store = Store("app_state.json")
        # self.bind(column_order=self.on_column_order_change)

    def update_column_order(self, columns: list):
        self.column_order = [col.id for col in columns]
        self.save_state("element_table", {"column_order": self.column_order})

    def save_state(self, index: str, data: dict):
        self._store.put(index, **data)

    def _load_state(self, item: str):
        if self._store.exists(item):
            data = self._store.get(item)
            self.column_order = data.get("column_order", [])
