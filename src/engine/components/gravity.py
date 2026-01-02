# TODO: Implement a more realistic velocity and collision damping algoritm to further improve visual feedback fidelity. In essence, prevent every rendered particle from falling at the same exact velocity.
import numpy as np
from numba import njit, prange

from .Constants import (
    COLLISION_DAMPING,
    TIME_STEP,
)
from .drag import apply_gravity_with_drag


@njit(fastmath=True)
def calculate_fall_distance(velocity_y, dt, max_distance=100):
    """Calculate how far a particle should fall this frame"""
    base_distance = abs(velocity_y * dt)
    return min(base_distance + 1, max_distance)


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
    y decreases downward, check cell below at y-1.
    """
    for particle_idx in sorted_active_indices:
        x = int(positions_x[particle_idx])
        y = int(positions_y[particle_idx])
        below_y = y - 1

        if 0 <= x < spatial_grid.shape[0] and 0 <= below_y < spatial_grid.shape[1]:
            collision_idx = spatial_grid[x, below_y]
            if collision_idx >= 0 and collision_idx != particle_idx:
                m1 = masses[particle_idx]
                m2 = masses[collision_idx]
                v1y = velocities_y[particle_idx]
                v2y = velocities_y[collision_idx]
                total = m1 + m2

                if total > 0:
                    # Calculate relative velocity (impact speed)
                    relative_v = v1y - v2y

                    # If particles moving together, no collision
                    if abs(relative_v) < 0.5:
                        continue

                    # Elastic collision
                    new_v1y = ((m1 - m2) * v1y + 2 * m2 * v2y) / total
                    new_v2y = ((m2 - m1) * v2y + 2 * m1 * v1y) / total

                    # Apply damping based on impact severity
                    impact_factor = min(abs(relative_v) / 10.0, 1.0)  # 0-1 scale
                    damping = 1.0 - (1.0 - COLLISION_DAMPING) * impact_factor

                    velocities_y[particle_idx] = new_v1y * damping
                    velocities_y[collision_idx] = new_v2y * damping


# @njit(fastmath=True)
# def calculate_position_changes(
#     positions_x,
#     positions_y,
#     velocities_y,
#     active_mask,
#     spatial_grid,
#     grid_height,
#     dt,
#     movement_array,
# ):
#     """
#     Calculate where particles want to move without actually moving them.
#     Returns arrays of old and new positions for particles that can move.
#     """
#     n = len(positions_x)
#     gx = spatial_grid.shape[0]
#     gy = spatial_grid.shape[1]
#
#     for i in prange(n):
#         if not active_mask[i]:
#             continue
#
#         vy = velocities_y[i]
#         if vy >= 0.0:  # Not moving downward
#             continue
#
#         x = int(positions_x[i])
#         y = int(positions_y[i])
#
#         if x < 0 or x >= gx or y < 0 or y >= gy:
#             continue
#
#         if y <= 0:  # Already at bottom
#             velocities_y[i] = 0.0
#             continue
#
#         fall_distance = calculate_fall_distance(vy, dt)
#         final_y = y
#
#         # Find how far particle can actually fall
#         for step in range(1, int(fall_distance + 1)):
#             test_y = y - step
#             if test_y <= 0 or test_y >= gy:
#                 break
#             if spatial_grid[x, test_y] == -1:
#                 final_y = test_y
#             else:
#                 break
#
#         if final_y != y:
#             movement_array[i]["fall_old_x"] = x
#             movement_array[i]["fall_old_y"] = y
#             movement_array[i]["fall_new_x"] = x
#             movement_array[i]["fall_new_y"] = final_y
#             movement_array[i]["fall_mask"] = True
#
#             # Decelerate based on distance fallen
#             velocities_y[i] = max(0.0, velocities_y[i] - (final_y - y) * 0.1)
#         else:
#             # Blocked, apply collision damping
#             velocities_y[i] *= COLLISION_DAMPING
#
#     return movement_array
@njit(fastmath=True)
def calculate_position_changes(
    positions_x,
    positions_y,
    velocities_y,
    active_mask,
    spatial_grid,
    grid_height,
    dt,
    movement_array,
):
    n = len(positions_x)
    gx = spatial_grid.shape[0]
    gy = spatial_grid.shape[1]

    # RESET MASKS FIRST!
    for i in range(len(movement_array)):
        movement_array[i]["fall_mask"] = False

    for i in prange(n):
        if not active_mask[i]:
            continue

        vy = velocities_y[i]
        if vy >= 0.0:
            continue

        x = int(positions_x[i])
        y = int(positions_y[i])

        if x < 0 or x >= gx or y < 0 or y >= gy:
            continue

        if y == 0:  # At ground
            velocities_y[i] = 0.0
            continue

        fall_distance = calculate_fall_distance(vy, dt)
        final_y = y
        blocked_by_particle = False  # NEW FLAG

        # Find how far particle can actually fall
        for step in range(1, int(fall_distance + 1)):
            test_y = y - step
            if test_y < 0 or test_y >= gy:
                break
            if spatial_grid[x, test_y] == -1:
                final_y = test_y
            else:
                blocked_by_particle = True  # We hit something!
                break

        if final_y != y:
            # Particle successfully moved
            movement_array[i]["fall_old_x"] = x
            movement_array[i]["fall_old_y"] = y
            movement_array[i]["fall_new_x"] = x
            movement_array[i]["fall_new_y"] = final_y
            movement_array[i]["fall_mask"] = True

            # Very gentle air resistance
            velocities_y[i] *= 0.99

        elif blocked_by_particle:
            # Actually blocked by an obstacle
            # Apply damping only if moving fast enough to care
            if abs(velocities_y[i]) > 1.0:
                velocities_y[i] *= COLLISION_DAMPING
            else:
                velocities_y[i] = 0.0  # Come to rest

        # else: didn't move but not blocked - keep velocity for next frame!
        # This happens when velocity is too small to move 1 pixel in this frame

    return movement_array


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
    movement_array,
):
    n = len(positions_x)
    gx = grid_width
    gy = grid_height

    # Reset masks
    for i in range(len(movement_array)):
        movement_array[i]["slide_mask"] = False

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

        # NEW: Only try to slide if particle has downward momentum
        # Don't slide if at rest!
        if abs(velocities_y[i]) < 0.5:
            continue  # Particle is resting, don't slide

        # If cell below is occupied, try diagonals
        if spatial_grid[x, below_y] != -1:
            # Determine slide direction preference
            dirs = (-1, 1)

            if velocities_x[i] > 0.1:
                dirs = (1, -1)
            elif velocities_x[i] < -0.1:
                dirs = (-1, 1)
            else:
                # NEW: Use position-based pseudo-random instead of index
                # This distributes particles more naturally
                if (x + y) % 2 == 0:  # Based on grid position, not particle ID
                    dirs = (1, -1)
                else:
                    dirs = (-1, 1)

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
                    movement_array[i]["slide_old_x"] = x
                    movement_array[i]["slide_old_y"] = y
                    movement_array[i]["slide_new_x"] = nx
                    movement_array[i]["slide_new_y"] = ny
                    movement_array[i]["slide_mask"] = True

                    # Update velocities with gentler damping
                    velocities_x[i] = (velocities_x[i] + dx * 0.3) * 0.95
                    velocities_y[i] *= 0.95  # Keep most downward momentum
                    moved = True
                    break

            if not moved:
                # Couldn't slide either - reduce velocity slightly
                velocities_y[i] *= 0.9

    return movement_array


# @njit(fastmath=True)
# def apply_position_changes(
#     positions_x,
#     positions_y,
#     spatial_grid,
#     old_x,
#     old_y,
#     new_x,
#     new_y,
#     change_mask,
# ):
#     """
#     Apply calculated position changes to the spatial grid and position arrays.
#     """
#     # First pass: clear old positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             if (
#                 0 <= old_x < spatial_grid.shape[0]
#                 and 0 <= old_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[old_x, old_y] = -1
#
#     # Second pass: set new positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             if (
#                 0 <= new_x < spatial_grid.shape[0]
#                 and 0 <= new_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[new_x, new_y] = i
#                 positions_x[i] = new_x
#                 positions_y[i] = new_y
@njit(fastmath=True)
def apply_position_changes(
    x,
    y,  # particle position arrays (modified in place)
    spatial_grid,  # 2D int32 array
    old_x,
    old_y,  # proposed old positions (per particle)
    new_x,
    new_y,  # proposed new positions
    mask,  # bool array: which particles to move
):
    moved_count = 0
    n = len(x)

    for i in prange(n):
        if mask[i]:
            ox = old_x[i]
            oy = old_y[i]
            nx = new_x[i]
            ny = new_y[i]

            # Clear old position in grid
            spatial_grid[ox, oy] = -1

            # Update particle position
            x[i] = nx
            y[i] = ny

            # Set new position in grid (store particle index)
            spatial_grid[nx, ny] = i

            moved_count += 1

    return moved_count  # optional: return how many actually moved


def update_gravity(simulation_grid, particles_buffer, dt=TIME_STEP):
    """Main physics update function: gravity, collisions, falling, sliding, and position updates"""
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

    fall_movement = particles_buffer["fall_movement"]
    slide_movement = particles_buffer["slide_movement"]

    stats = {
        "particles_moved": int(active_count),
        "particles_fell": 0,
        "particles_slid": 0,
    }

    # 1. Apply gravity (and drag) to vertical velocities
    apply_gravity_with_drag(
        particles["velocity_y"][:particle_count],
        particles["mass"][:particle_count],
        particles["drag_coeff"][:particle_count],
        active_mask,
        dt,
    )

    # 2. Resolve collisions (bottom-up for stability/determinism)
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

    # 3. Calculate all movement components
    particles_fell = calculate_position_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        particles["velocity_y"][:particle_count],
        active_mask,
        spatial_grid,
        simulation_grid.grid_height,
        dt,
        fall_movement[:particle_count],  # preserve any previous state if needed
    )

    # 4.
    particles_slid = calculate_slide_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        particles["velocity_x"][:particle_count],
        particles["velocity_y"][:particle_count],
        active_mask,
        spatial_grid,
        simulation_grid.grid_width,
        simulation_grid.grid_height,
        particles["state"][:particle_count],
        slide_movement[:particle_count],
    )
    particles_moved = 0
    # # 5. Apply particles_moved
    # apply_position_changes(
    #     particles["x"][:particle_count],
    #     particles["y"][:particle_count],
    #     spatial_grid,
    #     particles_fell["fall_old_x"],
    #     particles_fell["fall_old_y"],
    #     particles_fell["fall_new_x"],
    #     particles_fell["fall_new_y"],
    #     particles_fell["fall_mask"],
    # )
    #
    # apply_position_changes(
    #     particles["x"][:particle_count],
    #     particles["y"][:particle_count],
    #     spatial_grid,
    #     particles_slid["slide_old_x"],
    #     particles_slid["slide_old_y"],
    #     particles_slid["slide_new_x"],
    #     particles_slid["slide_new_y"],
    #     particles_slid["slide_mask"],
    # )
    #
    # Apply falling movements
    apply_position_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        spatial_grid,
        fall_movement["fall_old_x"][:particle_count],
        fall_movement["fall_old_y"][:particle_count],
        fall_movement["fall_new_x"][:particle_count],
        fall_movement["fall_new_y"][:particle_count],
        fall_movement["fall_mask"][:particle_count],
    )

    # Apply sliding movements
    apply_position_changes(
        particles["x"][:particle_count],
        particles["y"][:particle_count],
        spatial_grid,
        slide_movement["slide_old_x"][:particle_count],
        slide_movement["slide_old_y"][:particle_count],
        slide_movement["slide_new_x"][:particle_count],
        slide_movement["slide_new_y"][:particle_count],
        slide_movement["slide_mask"][:particle_count],
    )

    # Count actual movements
    particles_fell_count = int(np.sum(fall_movement["fall_mask"][:particle_count]))
    particles_slid_count = int(np.sum(slide_movement["slide_mask"][:particle_count]))

    return {
        "particles_moved": particles_fell_count + particles_slid_count,
        "particles_fell": particles_fell_count,
        "particles_slid": particles_slid_count,
    }
