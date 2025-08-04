# pyright: reportAttributeAccessIssue=false
import logging
from app.ui.elements import load_elements

from kivy.app import App
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.properties import (
    ObjectProperty,
)
from app.ui.modals.ScrollableTable import ScrollableTable

ELEMENTS = load_elements()


class PickerModal(ModalView):
    _cols = ObjectProperty({})

    def __init__(self, **kwargs):
        kwargs.setdefault("background", "")
        kwargs.setdefault("background_color", [0, 0, 0, 0])
        kwargs.setdefault("overlay_color", [0, 0, 0, 0.5])
        super().__init__(**kwargs)
        self.bind(on_parent=self._setup_scrollable_table)
        super(PickerModal, self).__init__(**kwargs)
        self._cols = {}

        # Replace the original table with our scrollable table
        # This assumes your KV file has an id called 'picker_table'
        self._setup_scrollable_table()

    def _setup_scrollable_table(self):
        """Replace the original table with our scrollable version"""
        if hasattr(self.ids, "picker_table_container"):
            # If you have a container in your KV file
            container = self.ids.picker_table_container
            container.clear_widgets()

            self.scrollable_table = ScrollableTable()
            container.add_widget(self.scrollable_table)
        else:
            logging.warning(
                "picker_table_container not found, using fallback container"
            )
            container = BoxLayout()  # Fallback container
            self.add_widget(container)

            self.scrollable_table = ScrollableTable()
            container.add_widget(self.scrollable_table)

    def on_pre_open(self):
        if not hasattr(self, "scrollable_table"):
            logging.error("Scrollable table not initialized")
            return

        self.scrollable_table.clear_table()
        self._cols = {}

        # Collect properties
        properties = set()
        for _, element in ELEMENTS.items():
            for k in element["intrinsic_properties"].keys():
                if k in [
                    "radius",
                    "phase_transitions",
                    "ignition_temperature",
                    "flame_temperature",
                    "burn_duration",
                    "oxygen_requirement",
                    "combustion_products",
                ]:
                    continue
                properties.add(k)

        properties = sorted(properties)
        self.scrollable_table.set_columns(len(properties))

        # Add headers
        for prop in properties:
            header = self.scrollable_table.add_header(prop, prop)
            self._cols[prop] = [header]

        # Add data rows
        for _, element in sorted(ELEMENTS.items()):
            for prop in properties:
                if prop == "color":
                    color_value = element["intrinsic_properties"].get(
                        prop, [0.74, 0.72, 0.42, 1.0]
                    )
                    if not isinstance(color_value, list) or len(color_value) != 4:
                        logging.warning(
                            f"Invalid color for element {element['intrinsic_properties']['id']}: {color_value}"
                        )
                        color_value = [0.74, 0.72, 0.42, 1.0]

                    label = Factory.PickerLabel(
                        text="",
                        background_color=color_value,
                        size_hint=(None, None),
                        height=30,
                    )
                else:
                    value = element["intrinsic_properties"].get(prop, "")
                    label = Factory.PickerLabel(
                        text=str(value), size_hint=(None, None), height=30
                    )

                self._cols[prop].append(label)
                self.scrollable_table.add_data_widget(label)

        app = App.get_running_app()
        if app and app.root.ids.simulation_cursor:
            app.root.ids.simulation_cursor._modal_open = True

        logging.debug(f"Table columns: {properties}")

    def on_pre_dismiss(self):
        app = App.get_running_app()
        if app and app.root.ids.simulation_cursor:
            app.root.ids.simulation_cursor._modal_open = False

    def reorder_columns(self, from_column, to_column):
        """Reorder columns with proper visual feedback"""
        if from_column == to_column:
            return

        properties = list(self._cols.keys())

        if from_column not in properties or to_column not in properties:
            logging.error(f"Invalid columns: {from_column}, {to_column}")
            return

        from_index = properties.index(from_column)
        to_index = properties.index(to_column)

        # Reorder the properties list
        properties.insert(to_index, properties.pop(from_index))

        # Rebuild the columns dictionary
        new_cols = {}
        for prop in properties:
            new_cols[prop] = self._cols[prop]

        self._cols.clear()
        self._cols.update(new_cols)

        # Rebuild the table
        self._rebuild_table(properties)

        logging.debug(f"Reordered columns: {properties}")

    def _rebuild_table(self, properties):
        """Rebuild the entire table with new column order"""
        self.scrollable_table.clear_table()
        self.scrollable_table.set_columns(len(properties))

        # Re-add headers in new order
        new_headers = []
        for prop in properties:
            header = self.scrollable_table.add_header(prop, prop)
            # Update the _cols reference
            self._cols[prop][0] = header
            new_headers.append(header)

        # Re-add data in new order
        num_elements = len(self._cols[properties[0]]) - 1  # -1 for header

        for row in range(1, num_elements + 1):  # Skip header row
            for prop in properties:
                widget = self._cols[prop][row]
                self.scrollable_table.add_data_widget(widget)

    def on_touch_down(self, touch):
        """Handle element selection"""
        if self.collide_point(*touch.pos) and hasattr(self, "scrollable_table"):
            # Check if clicking on a data cell for element selection
            for prop, widgets in self._cols.items():
                if prop == "id":  # Only handle selection on ID column
                    for widget in widgets[1:]:  # Skip header
                        if widget.collide_point(*touch.pos) and widget.text:
                            element_id = widget.text
                            app = App.get_running_app()
                            if app is not None and (
                                hasattr(app.root, "ids")
                                and "simulation_cursor" in app.root.ids
                            ):
                                app.root.ids.simulation_cursor.selected_element = (
                                    element_id
                                )
                                logging.debug(f"Selected element: {element_id}")
                                self.dismiss()
                                return True

        return super(PickerModal, self).on_touch_down(touch)
