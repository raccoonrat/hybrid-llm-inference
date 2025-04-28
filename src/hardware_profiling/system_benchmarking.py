class SystemBenchmarking:
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.profiler = None 