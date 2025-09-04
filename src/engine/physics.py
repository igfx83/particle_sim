# pyright: reportAttributeAccessIssue=false
from numba import njit, prange
import numpy as np


@njit(parallel=True, cache=True)
def update_gravity_and_fall_numba(
    active_indices,
    velocities_y,
    x_coords,
    y_coords,
    spatial_grid,
    grid_width,
    grid_height,
    states,
    dt,
    g=1.0,
):
    """Apply gravity and handle falling particles with numba acceleration"""
    n_particles = len(active_indices)
    can_fall = np.zeros(n_particles, dtype=np.bool8)

    # Apply gravity to all active particles
    for i in prange(n_particles):
        idx = active_indices[i]
        velocities_y[idx] -= g * dt

    # Check which solid particles can fall (state == 0)
    for i in prange(n_particles):
        idx = active_indices[i]
        if states[idx] != 0:  # Only solids fall directly
            continue

        x = x_coords[idx]
        y = y_coords[idx]

        # Bounds check
        if y <= 0 or y >= grid_height or x < 0 or x >= grid_width:
            continue

        below_y = y - 1
        if below_y >= 0 and spatial_grid[x, below_y] == -1:
            can_fall[i] = True

    return can_fall


@njit(parallel=True, cache=True)
def apply_particle_movement_numba(
    particle_indices,
    x_coords,
    y_coords,
    spatial_grid,
    velocities_y,
    dt,
    grid_width,
    grid_height,
):
    """Apply movement to particles that can fall"""
    n_particles = len(particle_indices)
    moved_particles = np.zeros(n_particles, dtype=np.int32)
    n_moved = 0

    for i in range(n_particles):
        idx = particle_indices[i]
        old_x = x_coords[idx]
        old_y = y_coords[idx]

        # Calculate fall distance based on velocity
        fall_dist = max(1, int(abs(velocities_y[idx] * dt)))
        new_y = max(0, old_y - fall_dist)

        if new_y < old_y and new_y >= 0:
            # Clear old position
            spatial_grid[old_x, old_y] = -1

            # Set new position
            y_coords[idx] = new_y
            spatial_grid[old_x, new_y] = idx

            # Dampen velocity on impact
            velocities_y[idx] *= 0.5

            moved_particles[n_moved] = idx
            n_moved += 1

    return moved_particles[:n_moved]


@njit(parallel=True, cache=True)
def update_temperatures_numba(
    active_indices,
    temperatures,
    burning,
    colors,
    dt,
    cooling_rate=0.1,
    burn_temp=800.0,
    ignition_temp=100.0,
):
    """Update particle temperatures and combustion states"""
    n_particles = len(active_indices)

    for i in prange(n_particles):
        idx = active_indices[i]
        temp = temperatures[idx]

        # Cool down hot particles
        if temp > 25.0:
            temperatures[idx] = temp - cooling_rate * dt

        # Handle combustion ignition
        if temp > ignition_temp and not burning[idx]:
            burning[idx] = True
            temperatures[idx] = burn_temp
            # Set burning color (orange)
            colors[idx, 0] = 1.0  # R
            colors[idx, 1] = 0.5  # G
            colors[idx, 2] = 0.0  # B
            colors[idx, 3] = 1.0  # A


@njit(parallel=True, cache=True)
def batch_heat_conduction_numba(
    conductive_positions,
    temperatures,
    spatial_grid,
    grid_width,
    grid_height,
    heat_conductivity,
    specific_heat,
    dt,
    min_temp_diff=0.1,
    conduction_rate=0.1,
):
    """Vectorized heat conduction between particles"""
    n_positions = len(conductive_positions)

    # Neighbor offsets for 4-connected grid
    neighbor_offsets = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)

    for i in prange(n_positions):
        x, y = conductive_positions[i]

        # Get particle index
        particle_idx = spatial_grid[x, y]
        if particle_idx == -1:
            continue

        particle_temp = temperatures[particle_idx]
        particle_conductivity = heat_conductivity[particle_idx]
        particle_heat_cap = specific_heat[particle_idx]

        # Check all 4 neighbors
        for j in range(4):
            dx, dy = neighbor_offsets[j]
            nx, ny = x + dx, y + dy

            # Bounds check
            if nx < 0 or nx >= grid_width or ny < 0 or ny >= grid_height:
                continue

            neighbor_idx = spatial_grid[nx, ny]
            if neighbor_idx == -1:
                continue

            neighbor_temp = temperatures[neighbor_idx]
            temp_diff = particle_temp - neighbor_temp

            # Skip tiny temperature differences
            if abs(temp_diff) < min_temp_diff:
                continue

            # Calculate heat transfer
            heat_transfer = particle_conductivity * conduction_rate * temp_diff * dt

            if abs(heat_transfer) > 0.01:
                neighbor_heat_cap = specific_heat[neighbor_idx]

                # Apply heat transfer
                temperatures[neighbor_idx] += heat_transfer / neighbor_heat_cap
                temperatures[particle_idx] -= heat_transfer / particle_heat_cap


@njit(cache=True)
def find_diagonal_slide_positions_numba(
    blocked_indices, x_coords, y_coords, spatial_grid, grid_width, grid_height
):
    """Find valid diagonal slide positions for blocked particles"""
    n_blocked = len(blocked_indices)
    slide_moves = np.empty(
        (n_blocked, 3), dtype=np.int32
    )  # [particle_idx, new_x, new_y]
    n_valid_moves = 0

    # Random direction choices for natural spreading
    directions = np.array([-1, 1], dtype=np.int32)

    for i in range(n_blocked):
        idx = blocked_indices[i]
        x = x_coords[idx]
        y = y_coords[idx]

        if y <= 0:
            continue

        below_y = y - 1
        moved = False

        # Try both diagonal directions randomly
        for dx in directions:
            new_x = x + dx

            # Bounds check
            if new_x < 0 or new_x >= grid_width or below_y < 0:
                continue

            # Check if diagonal position is free
            if spatial_grid[new_x, below_y] == -1:
                slide_moves[n_valid_moves, 0] = idx
                slide_moves[n_valid_moves, 1] = new_x
                slide_moves[n_valid_moves, 2] = below_y
                n_valid_moves += 1
                moved = True
                break

        if moved:
            continue

    return slide_moves[:n_valid_moves]


# Modified update method using numba functions
def update_with_numba(self, dt):
    """Optimized update method using numba-compiled functions"""

    # Periodic compaction
    if not hasattr(self, "frame_counter"):
        self.frame_counter = 0
    self.frame_counter += 1
    if self.frame_counter % 10 == 0:
        self.compact_arrays()

    # Get active particles
    active_mask = self.active[: self.particle_count]
    active_count = np.sum(active_mask)

    if active_count == 0:
        return

    active_indices = np.where(active_mask)[0]

    # === NUMBA OPTIMIZED PHYSICS ===

    # 1. Gravity and fall detection
    can_fall = update_gravity_and_fall_numba(
        active_indices,
        self.velocities_y,
        self.x_coords,
        self.y_coords,
        self.spatial_grid,
        self.grid_width,
        self.grid_height,
        self.states,
        dt,
    )

    # 2. Apply movement to falling particles
    solid_mask = self.states[: self.particle_count][active_mask] == 0
    solid_indices = active_indices[solid_mask]
    falling_particles = solid_indices[can_fall[solid_mask]]

    if len(falling_particles) > 0:
        moved_particles = apply_particle_movement_numba(
            falling_particles,
            self.x_coords,
            self.y_coords,
            self.spatial_grid,
            self.velocities_y,
            dt,
            self.grid_width,
            self.grid_height,
        )

    # 3. Handle diagonal slides for blocked particles
    blocked_indices = solid_indices[~can_fall[solid_mask]]
    if len(blocked_indices) > 0:
        # Sort by y descending for proper sliding
        sort_order = np.argsort(-self.y_coords[blocked_indices])
        blocked_indices = blocked_indices[sort_order]

        slide_moves = find_diagonal_slide_positions_numba(
            blocked_indices,
            self.x_coords,
            self.y_coords,
            self.spatial_grid,
            self.grid_width,
            self.grid_height,
        )

        # Apply slide moves
        for move in slide_moves:
            idx, new_x, new_y = move
            if self.move_particle(idx, new_x, new_y):
                self.velocities_y[idx] *= 0.8  # Dampen on slide

    # 4. Temperature and combustion updates
    update_temperatures_numba(
        active_indices, self.temperatures, self.burning, self.colors, dt
    )

    # 5. Heat conduction (if you have conductive particles)
    if hasattr(self, "_conductive_particles") and self._conductive_particles:
        conductive_positions = np.array(self._conductive_particles, dtype=np.int32)

        # Create heat conductivity and specific heat arrays for active particles
        heat_conductivity_array = np.ones(self.particle_count, dtype=np.float32) * 0.1
        specific_heat_array = np.ones(self.particle_count, dtype=np.float32)

        batch_heat_conduction_numba(
            conductive_positions,
            self.temperatures,
            self.spatial_grid,
            self.grid_width,
            self.grid_height,
            heat_conductivity_array,
            specific_heat_array,
            dt,
        )

    self.render()


# Additional numba utility function for neighbor finding
@njit(cache=True)
def get_neighbors_numba(x, y, grid_width, grid_height):
    """Fast neighbor coordinate generation"""
    neighbors = np.empty((4, 2), dtype=np.int32)
    count = 0

    # Check 4-connected neighbors
    offsets = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)

    for i in range(4):
        nx = x + offsets[i, 0]
        ny = y + offsets[i, 1]

        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            neighbors[count, 0] = nx
            neighbors[count, 1] = ny
            count += 1

    return neighbors[:count]


# Performance monitoring decorator
def time_function(func):
    """Decorator to measure function performance"""
    import time

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {(end - start) * 1000:.2f}ms")
        return result

    return wrapper
