def runge_kutta2_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + k1, y + h)

    # Update next value of y
    return x + (k1 + k2) / 2
