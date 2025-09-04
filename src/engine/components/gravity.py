import numpy as np
from numba import njit, prange

# Constants for physics calculations
GRAVITY_ACCELERATION = 4.0
TIME_STEP = 1.0 / 60.0
TERMINAL_VELOCITY = 20.0  # Maximum falling speed
VELOCITY_DAMPING = 0.98  # Air resistance factor
COLLISION_DAMPING = 0.7  # Energy loss on collision
MIN_VELOCITY_THRESHOLD = 0.01


@njit(fastmath=True)
def calculate_fall_distance(velocity_y, dt, max_distance=5):
    """Calculate how far a particle should fall this frame"""
    base_distance = int(abs(velocity_y * dt))
    return min(base_distance + 1, max_distance)


@njit(fastmath=True)
def apply_gravity_vectorized(velocity_y, active_mask, dt, g=GRAVITY_ACCELERATION):
    """Apply gravity to all active particles"""
    for i in prange(len(velocity_y)):
        if active_mask[i]:
            velocity_y[i] -= g * dt
            # Clamp to terminal velocity
            if velocity_y[i] > TERMINAL_VELOCITY:
                velocity_y[i] = TERMINAL_VELOCITY
            elif velocity_y[i] < -TERMINAL_VELOCITY:
                velocity_y[i] = -TERMINAL_VELOCITY

            # Apply damping
            velocity_y[i] *= VELOCITY_DAMPING

            # Zero out negligible velocities
            if abs(velocity_y[i]) < MIN_VELOCITY_THRESHOLD:
                velocity_y[i] = 0.0


@njit(fastmath=True)
def resolve_particle_collisions(
    positions_x,
    positions_y,
    velocities_x,
    velocities_y,
    spatial_grid,
    masses,
    sorted_active_indices,
):
    """
    Process particles bottom-up to resolve vertical collisions.
    y increases downward, check cell below at y+1.
    """
    for particle_idx in sorted_active_indices:
        x = int(positions_x[particle_idx])
        y = int(positions_y[particle_idx])
        below_y = y - 1  # Below is y+1 when y increases downward

        if 0 <= x < spatial_grid.shape[0] and 0 <= below_y < spatial_grid.shape[1]:
            collision_idx = spatial_grid[x, below_y]
            if collision_idx >= 0 and collision_idx != particle_idx:
                m1 = masses[particle_idx]
                m2 = masses[collision_idx]
                v1y = velocities_y[particle_idx]
                v2y = velocities_y[collision_idx]
                total = m1 + m2

                if total > 0:
                    # Elastic collision with damping
                    new_v1y = (
                        ((m1 - m2) * v1y + 2 * m2 * v2y) / total * COLLISION_DAMPING
                    )
                    new_v2y = (
                        ((m2 - m1) * v2y + 2 * m1 * v1y) / total * COLLISION_DAMPING
                    )
                    velocities_y[particle_idx] = new_v1y
                    velocities_y[collision_idx] = new_v2y


@njit(fastmath=True)
def calculate_position_changes(
    positions_x, positions_y, velocities_y, active_mask, spatial_grid, grid_height, dt
):
    """
    Calculate where particles want to move without actually moving them.
    Returns arrays of old and new positions for particles that can move.
    """
    n = len(positions_x)
    gx = spatial_grid.shape[0]
    gy = spatial_grid.shape[1]

    # Pre-allocate arrays for position changes
    old_positions_x = np.empty(n, dtype=np.int32)
    old_positions_y = np.empty(n, dtype=np.int32)
    new_positions_x = np.empty(n, dtype=np.int32)
    new_positions_y = np.empty(n, dtype=np.int32)
    fall_mask = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        if not active_mask[i]:
            continue

        vy = velocities_y[i]
        if vy >= 0.0:  # Not moving downward
            continue

        x = int(positions_x[i])
        y = int(positions_y[i])

        if x < 0 or x >= gx or y < 0 or y >= gy:
            continue

        if y <= 0:  # Already at bottom
            velocities_y[i] = 0.0
            continue

        fall_distance = calculate_fall_distance(vy, dt, max_distance=5)
        final_y = y

        # Find how far particle can actually fall
        for step in prange(1, fall_distance + 1):
            test_y = y - step
            if test_y <= 0 or test_y >= gy:
                break
            if spatial_grid[x, test_y] == -1:
                final_y = test_y
            else:
                break

        if final_y != y:
            old_positions_x[i] = x
            old_positions_y[i] = y
            new_positions_x[i] = x
            new_positions_y[i] = final_y
            fall_mask[i] = True

            # Decelerate based on distance fallen
            velocities_y[i] = max(0.0, velocities_y[i] - (final_y - y) * 0.1)
        else:
            # Blocked, apply collision damping
            velocities_y[i] *= COLLISION_DAMPING

    return old_positions_x, old_positions_y, new_positions_x, new_positions_y, fall_mask


@njit(fastmath=True)
def calculate_slide_changes(
    positions_x,
    positions_y,
    velocities_x,
    velocities_y,
    active_mask,
    spatial_grid,
    grid_width,
    grid_height,
    particle_states,
):
    """
    Calculate sliding movements without actually moving particles.
    """
    n = len(positions_x)
    gx = grid_width
    gy = grid_height

    old_positions_x = np.empty(n, dtype=np.int32)
    old_positions_y = np.empty(n, dtype=np.int32)
    new_positions_x = np.empty(n, dtype=np.int32)
    new_positions_y = np.empty(n, dtype=np.int32)
    slide_mask = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        if not active_mask[i]:
            continue
        if particle_states[i] != 0:  # Only solids slide
            continue

        x = int(positions_x[i])
        y = int(positions_y[i])

        if x < 0 or x >= gx or y < 0 or y >= gy:
            continue

        below_y = y - 1
        if below_y < 0:
            continue

        # If cell below is occupied, try diagonals
        if spatial_grid[x, below_y] != -1:
            # Determine slide direction preference
            dirs = (-1, 1)
            if velocities_x[i] > 0.1:
                dirs = (1, -1)
            elif velocities_x[i] < -0.1:
                dirs = (-1, 1)
            else:
                if i % 2 == 0:
                    dirs = (1, -1)

            moved = False
            for dx in dirs:
                nx = x + dx
                ny = y - 1
                if (
                    nx >= 0
                    and nx < gx
                    and ny >= 0
                    and ny < gy
                    and spatial_grid[nx, ny] == -1
                ):
                    old_positions_x[i] = x
                    old_positions_y[i] = y
                    new_positions_x[i] = nx
                    new_positions_y[i] = ny
                    slide_mask[i] = True

                    # Update velocities
                    velocities_x[i] += dx * 0.5
                    velocities_y[i] *= 0.9
                    moved = True
                    break

            if not moved:
                velocities_y[i] = 0.0

    return (
        old_positions_x,
        old_positions_y,
        new_positions_x,
        new_positions_y,
        slide_mask,
    )


@njit(fastmath=True)
def apply_position_changes(
    positions_x,
    positions_y,
    spatial_grid,
    old_pos_x,
    old_pos_y,
    new_pos_x,
    new_pos_y,
    change_mask,
):
    """
    Apply calculated position changes to the spatial grid and position arrays.
    """
    # First pass: clear old positions
    for i in prange(len(change_mask)):
        if change_mask[i]:
            old_x, old_y = old_pos_x[i], old_pos_y[i]
            if (
                0 <= old_x < spatial_grid.shape[0]
                and 0 <= old_y < spatial_grid.shape[1]
            ):
                spatial_grid[old_x, old_y] = -1

    # Second pass: set new positions
    for i in prange(len(change_mask)):
        if change_mask[i]:
            new_x, new_y = new_pos_x[i], new_pos_y[i]
            if (
                0 <= new_x < spatial_grid.shape[0]
                and 0 <= new_y < spatial_grid.shape[1]
            ):
                spatial_grid[new_x, new_y] = i
                positions_x[i] = new_x
                positions_y[i] = new_y


def update_gravity(simulation_grid, particles_buffer, dt=TIME_STEP):
    """Main physics update function with proper separation of concerns"""
    particle_count = particles_buffer["particle_count"]
    if particle_count == 0:
        return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}

    particles = particles_buffer["particles"]
    spatial_grid = particles_buffer["spatial_grid"]

    active_mask = particles["active"][:particle_count]
    active_count = np.sum(active_mask)

    if active_count == 0:
        return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}

    active_indices = np.where(active_mask)[0]

    # 1. Apply gravity to velocities
    apply_gravity_vectorized(particles["velocity_y"][:particle_count], active_mask, dt)

    # 2. Handle collisions (sorted for deterministic bottom-up processing)
    sorted_indices = np.argsort(particles["y"][:particle_count][active_mask])
    sorted_active_indices = active_indices[sorted_indices]

    resolve_particle_collisions(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        particles["velocity_x"][:particle_count],
        particles["velocity_y"][:particle_count],
        spatial_grid,
        particles["mass"][:particle_count],
        sorted_active_indices,
    )

    # 3. Calculate falling movements
    fall_old_x, fall_old_y, fall_new_x, fall_new_y, fall_mask = (
        calculate_position_changes(
            particles["x"][:particle_count],
            particles["y"][:particle_count],
            particles["velocity_y"][:particle_count],
            active_mask,
            spatial_grid,
            simulation_grid.grid_height,
            dt,
        )
    )

    # 4. Calculate sliding movements
    slide_old_x, slide_old_y, slide_new_x, slide_new_y, slide_mask = (
        calculate_slide_changes(
            particles["x"][:particle_count],
            particles["y"][:particle_count],
            particles["velocity_x"][:particle_count],
            particles["velocity_y"][:particle_count],
            active_mask,
            spatial_grid,
            simulation_grid.grid_width,
            simulation_grid.grid_height,
            particles["state"][:particle_count],
        )
    )

    # 5. Apply all position changes atomically
    apply_position_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        spatial_grid,
        fall_old_x,
        fall_old_y,
        fall_new_x,
        fall_new_y,
        fall_mask,
    )

    apply_position_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        spatial_grid,
        slide_old_x,
        slide_old_y,
        slide_new_x,
        slide_new_y,
        slide_mask,
    )
