from numba import njit, prange
from .Constants import GRAVITY_ACCELERATION


@njit(fastmath=True, parallel=True)
def apply_gravity_with_drag(velocity_y, masses, drag_coeffs, active_mask, dt):
    """
    Apply gravity with mass-dependent terminal velocity.

    Coordinate system: y decreases downward (negative y velocity = falling down)

    F_net = F_gravity - F_drag
    a = F_net / mass

    Drag must OPPOSE motion direction.
    """
    for i in prange(len(velocity_y)):
        if not active_mask[i]:
            continue

        mass = masses[i]

        # Safety check: avoid division by zero
        if mass < 1e-6:
            velocity_y[i] = 0.0
            continue

        # Current velocity (negative = falling downward)
        v = velocity_y[i]

        # Gravity acceleration
        # This makes particles accelerate downward (velocity becomes more negative)
        a_gravity = -GRAVITY_ACCELERATION

        # Formula: a_drag = -(k/m) * v² * sign(v)
        # Simplified: a_drag = -(k/m) * v * |v|
        if abs(v) > 1e-6:  # Only apply drag if moving
            a_drag = -drag_coeffs[i] * v * abs(v)
        else:
            a_drag = 0.0

        # Net acceleration
        a_net = a_gravity + a_drag

        # Update velocity
        new_v = v + a_net * dt

        # Clamp to reasonable values to prevent numerical instability
        if abs(new_v) < 1e-6:
            velocity_y[i] = 0.0
        elif new_v < -500.0:  # Terminal velocity cap (falling)
            velocity_y[i] = -500.0
        elif new_v > 500.0:  # Terminal velocity cap (rising)
            velocity_y[i] = 500.0
        else:
            velocity_y[i] = new_v
