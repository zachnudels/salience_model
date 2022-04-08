class Unit:
    """
    Class Representing one unit (with some feature preference) of a node in V1, V2, V4 of the model
    Each unit has four cells (represented as floats) interacting with one another as well as other units
    in higher and lower regions
    """

    def __init__(self,
                 a: float,
                 c1: float,
                 e1: float,
                 e2: float,
                 k: float,
                 g1: float,
                 g2: float):
        # PARAMETERS
        self.a = a
        self.c1 = c1
        self.e1 = e1
        self.e2 = e2
        self.k = k
        self.g1 = g1
        self.g2 = g2

        self.v = 0.0  # Input cell
        self.d = 0.0  # Feedback cell (region filling)
        self.u = 0.0  # Inhibitory cell
        self.w = 0.0  # Center-surround interactions (boundary detection)

    def v_dot(self, feedforward_signal: float):
        leak_conductance = -self.g1 * self.u * self.v
        modulated_driving_input = self.a * feedforward_signal * (1 + self.k * self.d) * (self.e1 - self.v)
        boundary_detection = self.g2 * self.w * (self.e2 - self.v)
        v_dot = leak_conductance + modulated_driving_input + boundary_detection
        return v_dot / self.c1
        
    def update(self, feedforward_signal: float, timestep: float):
        self.v += self.v_dot(feedforward_signal) * timestep

