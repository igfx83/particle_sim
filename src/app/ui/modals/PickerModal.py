# pyright: reportAttributeAccessIssue=false
import logging
from kivy.app import App
from kivy.properties import ObjectProperty
from app.ui.elements import load_elements
from app.ui.modals.ScrollableTable import ScrollableTable
from kivy.uix.modalview import ModalView

ELEMENTS = load_elements()


class PickerModal(ModalView):
    element_table = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(PickerModal, self).__init__(**kwargs)
        self.element_table = ScrollableTable()
        self.app = App.get_running_app()

    def on_touch_move(self, touch):
        logging.debug(
            f"Original Pos: {touch.ox}\nLast Pos: {touch.px}\nCurrent Pos: {touch.x}\nDelta Pos: {touch.dx}"
        )
        return super().on_touch_move(touch)

    def on_pre_open(self):
        if self.app and self.app.root.ids.simulation_cursor:
            self.app.root.ids.simulation_cursor._modal_open = True
            self.ids.picker_modal_panel.add_widget(self.element_table)

    def on_pre_dismiss(self):
        if self.app and self.app.root.ids.simulation_cursor:
            self.app.root.ids.simulation_cursor._modal_open = False
