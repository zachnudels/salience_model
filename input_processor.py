class InputProcessor:
    """
    Class Representing one unit (with some feature preference) of a node in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """
    def __init__(self, feature_pref: int, x: int, y: int):
        self.feature_pref = feature_pref
        self.x = x
        self.y = y
        self.v = 0.0
        self.w = 0.0

    def v_dot(self, activity: float) -> float:
        excitatory = activity * (10 - self.v)
        inhibitory = 2 * self.w ** 2 * self.v
        v_dot = excitatory - inhibitory
        return v_dot

    def w_dot(self) -> float:
        excitatory = self.v * (25 - self.w)
        inhibitory = self.w
        return (excitatory - inhibitory) / 5

    def update(self, _input: int, timestep: float) -> None:
        if _input == self.feature_pref:
            activity = 1
        else:
            activity = 0

        v_dot = self.v_dot(activity)
        w_dot = self.w_dot()

        self.v += v_dot * timestep
        self.w += w_dot * timestep

        if self.v < 0:
            self.v = 0
        if self.w < 0:
            self.w = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"({self.v}, {self.w})"