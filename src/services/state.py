from kivy.event import EventDispatcher
from kivy.storage.jsonstore import JsonStore as Store
from kivy.properties import ObjectProperty


class AppState(EventDispatcher):
    """Singleton class to manage application state."""

    element_table = ObjectProperty([])

    def __init__(self):
        super().__init__()
        self._store = Store("app_state.json")
        self.bind(element_table=self.on_element_table_change)

    def on_element_table_change(self, instance, value):
        self.save_state("element_table", self.element_table)

    def save_state(self, index: str, data: dict):
        self._store.put(index, **data)

    def _load_state(self, item: str):
        if self._store.exists(item):
            data = self._store.get(item)
            self.element_table = data.get("element_table", {})
