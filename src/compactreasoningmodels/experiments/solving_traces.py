class SolvingTraceStore:
    def __init__(self, dataloader):
        self.traces = {}
        self.dataloader = dataloader

    def add_trace(self, solver, args, id, accuracy_threshold=0.95, steps_threshold=30):
        pass

    def add_trace_fixed_steps(self, solver, args, id, steps):
        pass

    def get_trace(self, id):
        pass

    def get_all_traces(self):
        pass

    def get_traces_product(self):
        pass