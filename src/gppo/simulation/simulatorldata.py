from types import SimpleNamespace


class SimulatorRLData(SimpleNamespace):
    def __init__(self, obj):
        super().__init__(**obj.__dict__)
