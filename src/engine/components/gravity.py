# import numpy as np
# from numba import njit, prange
#
# # Constants for physics calculations
# GRAVITY_ACCELERATION = 150.0
# TIME_STEP = 1.0 / 60.0
# TERMINAL_VELOCITY = 500.0
# VELOCITY_DAMPING = 0.999
# COLLISION_DAMPING = 0.8
# MIN_VELOCITY_THRESHOLD = 0.01
#
#
# @njit(fastmath=True)
# def rebuild_spatial_grid_from_float(
#     spatial_grid, float_positions_x, float_positions_y, active_mask
# ):
#     """Rebuild spatial grid from float positions"""
#     # Clear the entire grid
#     spatial_grid.fill(-1)
#
#     # Rebuild from active particles
#     for i in range(len(active_mask)):
#         if active_mask[i]:
#             x = int(float_positions_x[i])
#             y = int(float_positions_y[i])
#
#             # Bounds check
#             if 0 <= x < spatial_grid.shape[0] and 0 <= y < spatial_grid.shape[1]:
#                 spatial_grid[x, y] = i
#
#
# @njit(fastmath=True)
# def apply_gravity_vectorized(velocity_y, active_mask, dt, g=GRAVITY_ACCELERATION):
#     """Apply gravity to all active particles"""
#     for i in prange(len(velocity_y)):
#         if active_mask[i]:
#             # Apply gravity acceleration
#             velocity_y[i] -= g * dt
#
#             # Clamp to terminal velocity
#             if velocity_y[i] < -TERMINAL_VELOCITY:
#                 velocity_y[i] = -TERMINAL_VELOCITY
#             elif velocity_y[i] > TERMINAL_VELOCITY:
#                 velocity_y[i] = TERMINAL_VELOCITY
#
#             # Apply minimal damping
#             if abs(velocity_y[i]) > MIN_VELOCITY_THRESHOLD:
#                 velocity_y[i] *= VELOCITY_DAMPING
#             else:
#                 velocity_y[i] = 0.0
#
#
# @njit(fastmath=True)
# def update_particles_unified(
#     float_positions_x,
#     float_positions_y,
#     velocities_x,
#     velocities_y,
#     active_mask,
#     spatial_grid,
#     dt,
# ):
#     """Update all particle positions using only float arrays"""
#     n = len(float_positions_x)
#     gx, gy = spatial_grid.shape
#
#     # Track which particles moved for grid updates
#     moved_particles = []
#     old_positions = []
#     new_positions = []
#
#     for i in prange(n):
#         if not active_mask[i]:
#             continue
#
#         # Current position
#         old_x = float_positions_x[i]
#         old_y = float_positions_y[i]
#         old_grid_x = int(old_x)
#         old_grid_y = int(old_y)
#
#         # Skip if out of bounds
#         if old_grid_x < 0 or old_grid_x >= gx or old_grid_y < 0 or old_grid_y >= gy:
#             continue
#
#         # Calculate new position from velocity
#         new_x = old_x + velocities_x[i] * dt
#         new_y = old_y + velocities_y[i] * dt
#
#         # Clamp to world bounds
#         new_x = max(0.0, min(float(gx - 1), new_x))
#         new_y = max(0.0, min(float(gy - 1), new_y))
#
#         new_grid_x = int(new_x)
#         new_grid_y = int(new_y)
#
#         # Check for collisions along the path (simplified for vertical movement)
#         collision_occurred = False
#         final_x = new_x
#         final_y = new_y
#
#         # For downward movement (negative velocity_y)
#         if velocities_y[i] < 0 and new_grid_y < old_grid_y:
#             # Check each grid cell we're moving through
#             for test_y in range(old_grid_y - 1, new_grid_y - 1, -1):
#                 if test_y < 0:
#                     final_y = 0.0
#                     velocities_y[i] = 0.0  # Hit ground
#                     collision_occurred = True
#                     break
#
#                 if test_y < gy and spatial_grid[new_grid_x, test_y] != -1:
#                     collision_idx = spatial_grid[new_grid_x, test_y]
#                     # Valid collision with another particle
#                     if (
#                         collision_idx >= 0
#                         and collision_idx < n
#                         and collision_idx != i
#                         and active_mask[collision_idx]
#                     ):
#                         # Stop just above the collision
#                         final_y = float(test_y + 1)
#                         velocities_y[i] *= COLLISION_DAMPING
#                         collision_occurred = True
#                         break
#                     else:
#                         # Invalid particle reference - clear it
#                         spatial_grid[new_grid_x, test_y] = -1
#
#         # Handle sliding for particles that can't fall
#         if collision_occurred and old_grid_y > 0:
#             # Try sliding left or right
#             slide_dirs = [-1, 1]
#             if velocities_x[i] > 0.1:
#                 slide_dirs = [1, -1]
#             elif velocities_x[i] < -0.1:
#                 slide_dirs = [-1, 1]
#
#             for dx in slide_dirs:
#                 slide_x = old_grid_x + dx
#                 slide_y = old_grid_y - 1
#
#                 if (
#                     slide_x >= 0
#                     and slide_x < gx
#                     and slide_y >= 0
#                     and slide_y < gy
#                     and spatial_grid[slide_x, slide_y] == -1
#                 ):
#                     # Can slide here
#                     final_x = float(slide_x)
#                     final_y = float(slide_y)
#                     velocities_x[i] += dx * 0.3
#                     velocities_y[i] *= 0.95
#                     break
#
#         # Update position
#         float_positions_x[i] = final_x
#         float_positions_y[i] = final_y
#
#         # Track grid changes
#         final_grid_x = int(final_x)
#         final_grid_y = int(final_y)
#
#         if old_grid_x != final_grid_x or old_grid_y != final_grid_y:
#             moved_particles.append(i)
#             old_positions.append((old_grid_x, old_grid_y))
#             new_positions.append((final_grid_x, final_grid_y))
#
#     return moved_particles, old_positions, new_positions
#
#
# @njit(fastmath=True)
# def update_spatial_grid(spatial_grid, moved_particles, old_positions, new_positions):
#     """Update spatial grid based on particle movements"""
#     gx, gy = spatial_grid.shape
#
#     # Clear old positions
#     for i, (old_x, old_y) in enumerate(old_positions):
#         if 0 <= old_x < gx and 0 <= old_y < gy:
#             particle_idx = moved_particles[i]
#             if spatial_grid[old_x, old_y] == particle_idx:
#                 spatial_grid[old_x, old_y] = -1
#
#     # Set new positions
#     for i, (new_x, new_y) in enumerate(new_positions):
#         if 0 <= new_x < gx and 0 <= new_y < gy:
#             particle_idx = moved_particles[i]
#             if spatial_grid[new_x, new_y] == -1:  # Only if cell is free
#                 spatial_grid[new_x, new_y] = particle_idx
#
#
# @njit(fastmath=True, parallel=True)
# def resolve_particle_collisions_unified(
#     float_positions_x,
#     float_positions_y,
#     velocities_x,
#     velocities_y,
#     spatial_grid,
#     masses,
#     active_mask,
# ):
#     """Handle particle-to-particle collisions using float positions"""
#     n = len(float_positions_x)
#     gx, gy = spatial_grid.shape
#
#     # Create sorted indices for bottom-up processing
#     active_indices = []
#     for i in range(n):
#         if active_mask[i]:
#             active_indices.append(i)
#
#     # Sort by y position (bottom first)
#     for i in range(len(active_indices)):
#         for j in range(i + 1, len(active_indices)):
#             idx1 = active_indices[i]
#             idx2 = active_indices[j]
#             if (
#                 float_positions_y[idx1] < float_positions_y[idx2]
#             ):  # Swap if idx1 is higher
#                 active_indices[i], active_indices[j] = idx2, idx1
#
#     for particle_idx in active_indices:
#         x = int(float_positions_x[particle_idx])
#         y = int(float_positions_y[particle_idx])
#         below_y = y - 1
#
#         if 0 <= x < gx and 0 <= below_y < gy:
#             collision_idx = spatial_grid[x, below_y]
#             if (
#                 collision_idx >= 0
#                 and collision_idx != particle_idx
#                 and collision_idx < n
#                 and active_mask[collision_idx]
#             ):
#                 m1 = masses[particle_idx]
#                 m2 = masses[collision_idx]
#                 v1y = velocities_y[particle_idx]
#                 v2y = velocities_y[collision_idx]
#                 total = m1 + m2
#
#                 if total > 0:
#                     # Elastic collision with damping
#                     new_v1y = (
#                         ((m1 - m2) * v1y + 2 * m2 * v2y) / total * COLLISION_DAMPING
#                     )
#                     new_v2y = (
#                         ((m2 - m1) * v2y + 2 * m1 * v1y) / total * COLLISION_DAMPING
#                     )
#                     velocities_y[particle_idx] = new_v1y
#                     velocities_y[collision_idx] = new_v2y
#
#
# def update_gravity_unified(simulation_grid, particles_buffer, dt=TIME_STEP):
#     """Unified physics update using only sub-pixel positions"""
#     particle_count = particles_buffer["particle_count"]
#     if particle_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     particles = particles_buffer["particles"]
#     spatial_grid = particles_buffer["spatial_grid"]
#
#     # Convert to unified float-based system
#     if "float_x" not in particles_buffer:
#         # Initialize from existing discrete positions
#         particles_buffer["float_x"] = particles["x"][:particle_count].astype(np.float64)
#         particles_buffer["float_y"] = particles["y"][:particle_count].astype(np.float64)
#         # Rebuild spatial grid to ensure consistency
#         rebuild_spatial_grid_from_float(
#             spatial_grid,
#             particles_buffer["float_x"][:particle_count],
#             particles_buffer["float_y"][:particle_count],
#             particles["active"][:particle_count],
#         )
#
#     # Ensure float arrays are properly sized
#     if len(particles_buffer["float_x"]) < particle_count:
#         new_size = len(particles["x"])
#         new_float_x = np.zeros(new_size, dtype=np.float64)
#         new_float_y = np.zeros(new_size, dtype=np.float64)
#
#         # Copy existing data
#         old_size = len(particles_buffer["float_x"])
#         new_float_x[:old_size] = particles_buffer["float_x"]
#         new_float_y[:old_size] = particles_buffer["float_y"]
#
#         # Initialize new particles from discrete positions
#         new_float_x[old_size:particle_count] = particles["x"][old_size:particle_count]
#         new_float_y[old_size:particle_count] = particles["y"][old_size:particle_count]
#
#         particles_buffer["float_x"] = new_float_x
#         particles_buffer["float_y"] = new_float_y
#
#     float_positions_x = particles_buffer["float_x"][:particle_count]
#     float_positions_y = particles_buffer["float_y"][:particle_count]
#     active_mask = particles["active"][:particle_count]
#
#     active_count = np.sum(active_mask)
#     if active_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     # 1. Apply gravity
#     apply_gravity_vectorized(particles["velocity_y"][:particle_count], active_mask, dt)
#
#     # 2. Handle collisions
#     resolve_particle_collisions_unified(
#         float_positions_x,
#         float_positions_y,
#         particles["velocity_x"][:particle_count],
#         particles["velocity_y"][:particle_count],
#         spatial_grid,
#         particles["mass"][:particle_count],
#         active_mask,
#     )
#
#     # 3. Update positions and handle collisions
#     moved_particles, old_positions, new_positions = update_particles_unified(
#         float_positions_x,
#         float_positions_y,
#         particles["velocity_x"][:particle_count],
#         particles["velocity_y"][:particle_count],
#         active_mask,
#         spatial_grid,
#         dt,
#     )
#
#     # 4. Update spatial grid
#     update_spatial_grid(spatial_grid, moved_particles, old_positions, new_positions)
#
#     # 5. Sync discrete positions for compatibility (optional - can be removed later)
#     for i in range(particle_count):
#         particles["x"][i] = int(float_positions_x[i])
#         particles["y"][i] = int(float_positions_y[i])
#
#     return {
#         "particles_moved": len(moved_particles),
#         "particles_fell": np.sum(
#             old_pos_y > new_pos_y
#         ),  # Count particles that moved down
#         "particles_slid": np.sum(
#             old_pos_x != new_pos_x
#         ),  # Count particles that moved horizontally
#     }
#
#
# # Drop-in replacement
# def update_gravity(simulation_grid, particles_buffer, dt=TIME_STEP):
#     """Drop-in replacement using unified position system"""
#     return update_gravity_unified(simulation_grid, particles_buffer, dt)


# import numpy as np
# from numba import njit, prange
#
# # Constants for physics calculations
# GRAVITY_ACCELERATION = 150.0
# TIME_STEP = 1.0 / 60.0
# TERMINAL_VELOCITY = 500.0
# VELOCITY_DAMPING = 0.999
# COLLISION_DAMPING = 0.8
# MIN_VELOCITY_THRESHOLD = 0.01
#
#
# @njit(fastmath=True)
# def apply_gravity_vectorized(velocity_y, active_mask, dt, g=GRAVITY_ACCELERATION):
#     """Apply gravity to all active particles"""
#     for i in prange(len(velocity_y)):
#         if active_mask[i]:
#             # Apply gravity acceleration
#             velocity_y[i] -= g * dt
#
#             # Clamp to terminal velocity
#             if velocity_y[i] < -TERMINAL_VELOCITY:
#                 velocity_y[i] = -TERMINAL_VELOCITY
#             elif velocity_y[i] > TERMINAL_VELOCITY:
#                 velocity_y[i] = TERMINAL_VELOCITY
#
#             # Apply minimal damping
#             if abs(velocity_y[i]) > MIN_VELOCITY_THRESHOLD:
#                 velocity_y[i] *= VELOCITY_DAMPING
#             else:
#                 velocity_y[i] = 0.0
#
#
# @njit(fastmath=True)
# def calculate_position_changes_smooth(
#     positions_x,
#     positions_y,
#     float_positions_x,  # NEW: Sub-pixel positions
#     float_positions_y,  # NEW: Sub-pixel positions
#     velocities_y,
#     active_mask,
#     spatial_grid,
#     grid_height,
#     dt,
# ):
#     """Calculate movements with sub-pixel precision"""
#     n = len(positions_x)
#     gx = spatial_grid.shape[0]
#     gy = spatial_grid.shape[1]
#
#     old_positions_x = np.empty(n, dtype=np.int32)
#     old_positions_y = np.empty(n, dtype=np.int32)
#     new_positions_x = np.empty(n, dtype=np.int32)
#     new_positions_y = np.empty(n, dtype=np.int32)
#     fall_mask = np.zeros(n, dtype=np.bool_)
#
#     for i in prange(n):
#         if not active_mask[i]:
#             continue
#
#         vy = velocities_y[i]
#         if vy >= 0.0:  # Not moving downward
#             continue
#
#         # Work with floating point positions
#         current_float_y = float_positions_y[i]
#         current_grid_x = int(positions_x[i])
#         current_grid_y = int(positions_y[i])
#
#         if (
#             current_grid_x < 0
#             or current_grid_x >= gx
#             or current_grid_y < 0
#             or current_grid_y >= gy
#         ):
#             continue
#
#         if current_grid_y <= 0:
#             velocities_y[i] = 0.0
#             continue
#
#         # Calculate new floating point position
#         new_float_y = current_float_y + vy * dt  # vy is negative for downward
#         new_grid_y = int(new_float_y)
#
#         # Clamp to bounds
#         new_grid_y = max(0, min(new_grid_y, gy - 1))
#         new_float_y = max(0.0, new_float_y)
#
#         # Check for collisions along the path
#         final_grid_y = current_grid_y
#         final_float_y = current_float_y
#         collision_occurred = False
#
#         # Test each grid cell the particle would pass through
#         start_y = current_grid_y
#         end_y = new_grid_y
#
#         if start_y != end_y:
#             # Check intermediate positions
#             for test_y in range(start_y - 1, end_y - 1, -1):
#                 if test_y < 0:
#                     final_grid_y = 0
#                     final_float_y = 0.0
#                     collision_occurred = True
#                     break
#                 if test_y >= gy:
#                     continue
#
#                 if spatial_grid[current_grid_x, test_y] != -1:
#                     # Collision detected - stop at the previous valid position
#                     final_grid_y = test_y + 1
#                     final_float_y = float(final_grid_y)
#                     collision_occurred = True
#                     break
#                 else:
#                     final_grid_y = test_y
#                     final_float_y = new_float_y
#         else:
#             # No grid boundary crossed, but update float position
#             final_float_y = new_float_y
#
#         # Update positions if there was movement
#         if final_grid_y != current_grid_y:
#             old_positions_x[i] = current_grid_x
#             old_positions_y[i] = current_grid_y
#             new_positions_x[i] = current_grid_x
#             new_positions_y[i] = final_grid_y
#             fall_mask[i] = True
#
#         # Always update the floating point position
#         float_positions_y[i] = final_float_y
#
#         # Handle velocity after collision
#         if collision_occurred:
#             if final_grid_y == 0:
#                 velocities_y[i] = 0.0  # Hit ground
#             else:
#                 velocities_y[i] *= COLLISION_DAMPING  # Hit particle
#
#     return old_positions_x, old_positions_y, new_positions_x, new_positions_y, fall_mask
#
#
# @njit(fastmath=True)
# def calculate_slide_changes_smooth(
#     positions_x,
#     positions_y,
#     float_positions_x,
#     float_positions_y,
#     velocities_x,
#     velocities_y,
#     active_mask,
#     spatial_grid,
#     grid_width,
#     grid_height,
#     particle_states,
#     dt,
# ):
#     """Calculate sliding movements with sub-pixel precision"""
#     n = len(positions_x)
#     gx = grid_width
#     gy = grid_height
#
#     old_positions_x = np.empty(n, dtype=np.int32)
#     old_positions_y = np.empty(n, dtype=np.int32)
#     new_positions_x = np.empty(n, dtype=np.int32)
#     new_positions_y = np.empty(n, dtype=np.int32)
#     slide_mask = np.zeros(n, dtype=np.bool_)
#
#     for i in prange(n):
#         if not active_mask[i]:
#             continue
#         if particle_states[i] != 0:  # Only solids slide
#             continue
#
#         x = int(positions_x[i])
#         y = int(positions_y[i])
#
#         if x < 0 or x >= gx or y < 0 or y >= gy:
#             continue
#
#         below_y = y - 1
#         if below_y < 0:
#             continue
#
#         # If cell below is occupied, try diagonals
#         if spatial_grid[x, below_y] != -1:
#             # Determine slide direction preference
#             dirs = (-1, 1)
#             if velocities_x[i] > 0.1:
#                 dirs = (1, -1)
#             elif velocities_x[i] < -0.1:
#                 dirs = (-1, 1)
#             else:
#                 if i % 2 == 0:
#                     dirs = (1, -1)
#
#             moved = False
#             for dx in dirs:
#                 nx = x + dx
#                 ny = y - 1
#                 if (
#                     nx >= 0
#                     and nx < gx
#                     and ny >= 0
#                     and ny < gy
#                     and spatial_grid[nx, ny] == -1
#                 ):
#                     old_positions_x[i] = x
#                     old_positions_y[i] = y
#                     new_positions_x[i] = nx
#                     new_positions_y[i] = ny
#                     slide_mask[i] = True
#
#                     # Update both integer and float positions
#                     float_positions_x[i] = float(nx)
#                     float_positions_y[i] = float(ny)
#
#                     # Update velocities
#                     velocities_x[i] += dx * 0.3
#                     velocities_y[i] *= 0.95
#                     moved = True
#                     break
#
#             if not moved:
#                 velocities_y[i] *= 0.8
#
#     return (
#         old_positions_x,
#         old_positions_y,
#         new_positions_x,
#         new_positions_y,
#         slide_mask,
#     )
#
#
# @njit(fastmath=True, parallel=True)
# def resolve_particle_collisions(
#     positions_x,
#     positions_y,
#     velocities_x,
#     velocities_y,
#     spatial_grid,
#     masses,
#     sorted_active_indices,
# ):
#     """Process particles bottom-up to resolve vertical collisions."""
#     for particle_idx in sorted_active_indices:
#         x = int(positions_x[particle_idx])
#         y = int(positions_y[particle_idx])
#         below_y = y - 1
#
#         if 0 <= x < spatial_grid.shape[0] and 0 <= below_y < spatial_grid.shape[1]:
#             collision_idx = spatial_grid[x, below_y]
#             if collision_idx >= 0 and collision_idx != particle_idx:
#                 m1 = masses[particle_idx]
#                 m2 = masses[collision_idx]
#                 v1y = velocities_y[particle_idx]
#                 v2y = velocities_y[collision_idx]
#                 total = m1 + m2
#
#                 if total > 0:
#                     # Elastic collision with damping
#                     new_v1y = (
#                         ((m1 - m2) * v1y + 2 * m2 * v2y) / total * COLLISION_DAMPING
#                     )
#                     new_v2y = (
#                         ((m2 - m1) * v2y + 2 * m1 * v1y) / total * COLLISION_DAMPING
#                     )
#                     velocities_y[particle_idx] = new_v1y
#                     velocities_y[collision_idx] = new_v2y
#
#
# @njit(fastmath=True)
# def apply_position_changes(
#     positions_x,
#     positions_y,
#     spatial_grid,
#     old_pos_x,
#     old_pos_y,
#     new_pos_x,
#     new_pos_y,
#     change_mask,
# ):
#     """Apply calculated position changes to the spatial grid and position arrays."""
#     # First pass: clear old positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             old_x, old_y = old_pos_x[i], old_pos_y[i]
#             if (
#                 0 <= old_x < spatial_grid.shape[0]
#                 and 0 <= old_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[old_x, old_y] = -1
#
#     # Second pass: set new positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             new_x, new_y = new_pos_x[i], new_pos_y[i]
#             if (
#                 0 <= new_x < spatial_grid.shape[0]
#                 and 0 <= new_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[new_x, new_y] = i
#                 positions_x[i] = new_x
#                 positions_y[i] = new_y
#
#
# def update_gravity(simulation_grid, particles_buffer, dt=TIME_STEP):
#     """Main physics update function with smooth sub-pixel movement"""
#     particle_count = particles_buffer["particle_count"]
#     if particle_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     particles = particles_buffer["particles"]
#     spatial_grid = particles_buffer["spatial_grid"]
#
#     # Initialize sub-pixel position arrays if they don't exist
#     if "float_x" not in particles_buffer:
#         particles_buffer["float_x"] = particles["x"][:particle_count].astype(np.float64)
#         particles_buffer["float_y"] = particles["y"][:particle_count].astype(np.float64)
#
#     float_positions_x = particles_buffer["float_x"]
#     float_positions_y = particles_buffer["float_y"]
#
#     active_mask = particles["active"][:particle_count]
#     active_count = np.sum(active_mask)
#
#     if active_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     active_indices = np.where(active_mask)[0]
#
#     # 1. Apply gravity to velocities
#     apply_gravity_vectorized(particles["velocity_y"][:particle_count], active_mask, dt)
#
#     # 2. Handle collisions
#     sorted_indices = np.argsort(particles["y"][:particle_count][active_mask])
#     sorted_active_indices = active_indices[sorted_indices]
#
#     resolve_particle_collisions(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         particles["velocity_x"][:particle_count],
#         particles["velocity_y"][:particle_count],
#         spatial_grid,
#         particles["mass"][:particle_count],
#         sorted_active_indices,
#     )
#
#     # 3. Calculate falling movements with sub-pixel precision
#     fall_old_x, fall_old_y, fall_new_x, fall_new_y, fall_mask = (
#         calculate_position_changes_smooth(
#             particles["x"][:particle_count],
#             particles["y"][:particle_count],
#             float_positions_x[:particle_count],
#             float_positions_y[:particle_count],
#             particles["velocity_y"][:particle_count],
#             active_mask,
#             spatial_grid,
#             simulation_grid.grid_height,
#             dt,
#         )
#     )
#
#     # 4. Calculate sliding movements
#     slide_old_x, slide_old_y, slide_new_x, slide_new_y, slide_mask = (
#         calculate_slide_changes_smooth(
#             particles["x"][:particle_count],
#             particles["y"][:particle_count],
#             float_positions_x[:particle_count],
#             float_positions_y[:particle_count],
#             particles["velocity_x"][:particle_count],
#             particles["velocity_y"][:particle_count],
#             active_mask,
#             spatial_grid,
#             simulation_grid.grid_width,
#             simulation_grid.grid_height,
#             particles["state"][:particle_count],
#             dt,
#         )
#     )
#
#     # 5. Apply all position changes atomically
#     apply_position_changes(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         spatial_grid,
#         fall_old_x,
#         fall_old_y,
#         fall_new_x,
#         fall_new_y,
#         fall_mask,
#     )
#
#     apply_position_changes(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         spatial_grid,
#         slide_old_x,
#         slide_old_y,
#         slide_new_x,
#         slide_new_y,
#         slide_mask,
#     )
#
#     return {
#         "particles_moved": np.sum(fall_mask) + np.sum(slide_mask),
#         "particles_fell": np.sum(fall_mask),
#         "particles_slid": np.sum(slide_mask),
#     }


# import numpy as np
# from numba import njit, prange
#
# # Constants for physics calculations
# GRAVITY_ACCELERATION = 150.0
# TIME_STEP = 1.0 / 60.0
# TERMINAL_VELOCITY = 500.0
# VELOCITY_DAMPING = 0.999  # Reduced damping so gravity can accumulate
# COLLISION_DAMPING = 0.8
# MIN_VELOCITY_THRESHOLD = 0.01
#
#
# @njit(fastmath=True)
# def calculate_fall_distance(velocity_y, dt):
#     """Calculate how far a particle should fall this frame - removed artificial limits"""
#     # Pure physics-based calculation
#     distance = abs(velocity_y * dt)
#     return max(1.0, distance)  # Minimum of 1 pixel movement
#
#
# @njit(fastmath=True)
# def apply_gravity_vectorized(velocity_y, active_mask, dt, g=GRAVITY_ACCELERATION):
#     """Apply gravity to all active particles"""
#     for i in prange(len(velocity_y)):
#         if active_mask[i]:
#             # Apply gravity acceleration (negative because y increases downward)
#             velocity_y[i] -= g * dt
#
#             # Clamp to terminal velocity
#             if velocity_y[i] < -TERMINAL_VELOCITY:
#                 velocity_y[i] = -TERMINAL_VELOCITY
#             elif velocity_y[i] > TERMINAL_VELOCITY:
#                 velocity_y[i] = TERMINAL_VELOCITY
#
#             # Apply minimal damping only if moving
#             if abs(velocity_y[i]) > MIN_VELOCITY_THRESHOLD:
#                 velocity_y[i] *= VELOCITY_DAMPING
#             else:
#                 velocity_y[i] = 0.0
#
#
# @njit(fastmath=True, parallel=True)
# def resolve_particle_collisions(
#     positions_x,
#     positions_y,
#     velocities_x,
#     velocities_y,
#     spatial_grid,
#     masses,
#     sorted_active_indices,
# ):
#     """Process particles bottom-up to resolve vertical collisions."""
#     for particle_idx in sorted_active_indices:
#         x = int(positions_x[particle_idx])
#         y = int(positions_y[particle_idx])
#         below_y = y - 1
#
#         if 0 <= x < spatial_grid.shape[0] and 0 <= below_y < spatial_grid.shape[1]:
#             collision_idx = spatial_grid[x, below_y]
#             if collision_idx >= 0 and collision_idx != particle_idx:
#                 m1 = masses[particle_idx]
#                 m2 = masses[collision_idx]
#                 v1y = velocities_y[particle_idx]
#                 v2y = velocities_y[collision_idx]
#                 total = m1 + m2
#
#                 if total > 0:
#                     # Elastic collision with damping
#                     new_v1y = (
#                         ((m1 - m2) * v1y + 2 * m2 * v2y) / total * COLLISION_DAMPING
#                     )
#                     new_v2y = (
#                         ((m2 - m1) * v2y + 2 * m1 * v1y) / total * COLLISION_DAMPING
#                     )
#                     velocities_y[particle_idx] = new_v1y
#                     velocities_y[collision_idx] = new_v2y
#
#
# @njit(fastmath=True)
# def calculate_position_changes(
#     positions_x, positions_y, velocities_y, active_mask, spatial_grid, grid_height, dt
# ):
#     """Calculate where particles want to move - FIXED VERSION"""
#     n = len(positions_x)
#     gx = spatial_grid.shape[0]
#     gy = spatial_grid.shape[1]
#
#     old_positions_x = np.empty(n, dtype=np.int32)
#     old_positions_y = np.empty(n, dtype=np.int32)
#     new_positions_x = np.empty(n, dtype=np.int32)
#     new_positions_y = np.empty(n, dtype=np.int32)
#     fall_mask = np.zeros(n, dtype=np.bool_)
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
#         # Calculate intended fall distance based on current velocity
#         fall_distance = calculate_fall_distance(vy, dt)
#         target_y = max(0, y - int(fall_distance))
#         final_y = y
#
#         # Find how far particle can actually fall
#         for test_y in range(y - 1, target_y - 1, -1):
#             if test_y < 0:
#                 final_y = 0
#                 break
#             if test_y >= gy:
#                 continue
#             if spatial_grid[x, test_y] == -1:
#                 final_y = test_y
#             else:
#                 break
#
#         if final_y != y:
#             old_positions_x[i] = x
#             old_positions_y[i] = y
#             new_positions_x[i] = x
#             new_positions_y[i] = final_y
#             fall_mask[i] = True
#
#             # CRITICAL FIX: Don't artificially reduce velocity based on collision
#             # Only reduce velocity if we hit the ground (y = 0)
#             if final_y == 0:
#                 velocities_y[i] = 0.0
#             # If blocked by another particle, apply collision damping but preserve most velocity
#             elif final_y > target_y:
#                 velocities_y[i] *= COLLISION_DAMPING
#         else:
#             # Completely blocked, apply stronger collision damping
#             velocities_y[i] *= COLLISION_DAMPING * 0.5
#
#     return old_positions_x, old_positions_y, new_positions_x, new_positions_y, fall_mask
#
#
# @njit(fastmath=True)
# def calculate_slide_changes(
#     positions_x,
#     positions_y,
#     velocities_x,
#     velocities_y,
#     active_mask,
#     spatial_grid,
#     grid_width,
#     grid_height,
#     particle_states,
# ):
#     """Calculate sliding movements without actually moving particles."""
#     n = len(positions_x)
#     gx = grid_width
#     gy = grid_height
#
#     old_positions_x = np.empty(n, dtype=np.int32)
#     old_positions_y = np.empty(n, dtype=np.int32)
#     new_positions_x = np.empty(n, dtype=np.int32)
#     new_positions_y = np.empty(n, dtype=np.int32)
#     slide_mask = np.zeros(n, dtype=np.bool_)
#
#     for i in prange(n):
#         if not active_mask[i]:
#             continue
#         if particle_states[i] != 0:  # Only solids slide
#             continue
#
#         x = int(positions_x[i])
#         y = int(positions_y[i])
#
#         if x < 0 or x >= gx or y < 0 or y >= gy:
#             continue
#
#         below_y = y - 1
#         if below_y < 0:
#             continue
#
#         # If cell below is occupied, try diagonals
#         if spatial_grid[x, below_y] != -1:
#             # Determine slide direction preference
#             dirs = (-1, 1)
#             if velocities_x[i] > 0.1:
#                 dirs = (1, -1)
#             elif velocities_x[i] < -0.1:
#                 dirs = (-1, 1)
#             else:
#                 if i % 2 == 0:
#                     dirs = (1, -1)
#
#             moved = False
#             for dx in dirs:
#                 nx = x + dx
#                 ny = y - 1
#                 if (
#                     nx >= 0
#                     and nx < gx
#                     and ny >= 0
#                     and ny < gy
#                     and spatial_grid[nx, ny] == -1
#                 ):
#                     old_positions_x[i] = x
#                     old_positions_y[i] = y
#                     new_positions_x[i] = nx
#                     new_positions_y[i] = ny
#                     slide_mask[i] = True
#
#                     # Update velocities - preserve more downward momentum
#                     velocities_x[i] += dx * 0.3
#                     velocities_y[i] *= 0.95  # Less velocity loss
#                     moved = True
#                     break
#
#             if not moved:
#                 # Reduce but don't zero out velocity - let gravity accumulate
#                 velocities_y[i] *= 0.8
#
#     return (
#         old_positions_x,
#         old_positions_y,
#         new_positions_x,
#         new_positions_y,
#         slide_mask,
#     )
#
#
# @njit(fastmath=True)
# def apply_position_changes(
#     positions_x,
#     positions_y,
#     spatial_grid,
#     old_pos_x,
#     old_pos_y,
#     new_pos_x,
#     new_pos_y,
#     change_mask,
# ):
#     """Apply calculated position changes to the spatial grid and position arrays."""
#     # First pass: clear old positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             old_x, old_y = old_pos_x[i], old_pos_y[i]
#             if (
#                 0 <= old_x < spatial_grid.shape[0]
#                 and 0 <= old_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[old_x, old_y] = -1
#
#     # Second pass: set new positions
#     for i in prange(len(change_mask)):
#         if change_mask[i]:
#             new_x, new_y = new_pos_x[i], new_pos_y[i]
#             if (
#                 0 <= new_x < spatial_grid.shape[0]
#                 and 0 <= new_y < spatial_grid.shape[1]
#             ):
#                 spatial_grid[new_x, new_y] = i
#                 positions_x[i] = new_x
#                 positions_y[i] = new_y
#
#
# def update_gravity(simulation_grid, particles_buffer, dt=TIME_STEP):
#     """Main physics update function with proper separation of concerns"""
#     particle_count = particles_buffer["particle_count"]
#     if particle_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     particles = particles_buffer["particles"]
#     spatial_grid = particles_buffer["spatial_grid"]
#
#     active_mask = particles["active"][:particle_count]
#     active_count = np.sum(active_mask)
#
#     if active_count == 0:
#         return {"particles_moved": 0, "particles_fell": 0, "particles_slid": 0}
#
#     active_indices = np.where(active_mask)[0]
#
#     # 1. Apply gravity to velocities
#     apply_gravity_vectorized(particles["velocity_y"][:particle_count], active_mask, dt)
#
#     # 2. Handle collisions (sorted for deterministic bottom-up processing)
#     sorted_indices = np.argsort(particles["y"][:particle_count][active_mask])
#     sorted_active_indices = active_indices[sorted_indices]
#
#     resolve_particle_collisions(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         particles["velocity_x"][:particle_count],
#         particles["velocity_y"][:particle_count],
#         spatial_grid,
#         particles["mass"][:particle_count],
#         sorted_active_indices,
#     )
#
#     # 3. Calculate falling movements
#     fall_old_x, fall_old_y, fall_new_x, fall_new_y, fall_mask = (
#         calculate_position_changes(
#             particles["x"][:particle_count],
#             particles["y"][:particle_count],
#             particles["velocity_y"][:particle_count],
#             active_mask,
#             spatial_grid,
#             simulation_grid.grid_height,
#             dt,
#         )
#     )
#
#     # 4. Calculate sliding movements
#     slide_old_x, slide_old_y, slide_new_x, slide_new_y, slide_mask = (
#         calculate_slide_changes(
#             particles["x"][:particle_count],
#             particles["y"][:particle_count],
#             particles["velocity_x"][:particle_count],
#             particles["velocity_y"][:particle_count],
#             active_mask,
#             spatial_grid,
#             simulation_grid.grid_width,
#             simulation_grid.grid_height,
#             particles["state"][:particle_count],
#         )
#     )
#
#     # 5. Apply all position changes atomically
#     apply_position_changes(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         spatial_grid,
#         fall_old_x,
#         fall_old_y,
#         fall_new_x,
#         fall_new_y,
#         fall_mask,
#     )
#
#     apply_position_changes(
#         particles["x"][:particle_count],
#         particles["y"][:particle_count],
#         spatial_grid,
#         slide_old_x,
#         slide_old_y,
#         slide_new_x,
#         slide_new_y,
#         slide_mask,
#     )
#
#     return {
#         "particles_moved": np.sum(fall_mask) + np.sum(slide_mask),
#         "particles_fell": np.sum(fall_mask),
#         "particles_slid": np.sum(slide_mask),
#     }
#

# TODO: Implement a more realistic velocity and collision damping algoritm to further improve visual feedback fidelity. In essence, prevent every rendered particle from falling at the same exact velocity.
import numpy as np
from numba import njit, prange

# Constants for physics calculations
GRAVITY_ACCELERATION = 150.0
TIME_STEP = 1.0 / 60.0
TERMINAL_VELOCITY = 500.0  # Maximum falling speed
VELOCITY_DAMPING = 1.0  # Air resistance factor
COLLISION_DAMPING = 1.0  # Energy loss on collision
MIN_VELOCITY_THRESHOLD = 0.01


@njit(fastmath=True, parallel=True)
def calculate_fall_distance(velocity_y, dt, max_distance=100):
    """Calculate how far a particle should fall this frame"""
    base_distance = abs(velocity_y * dt)
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


@njit(fastmath=True, parallel=True)
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

        fall_distance = calculate_fall_distance(vy, dt)
        final_y = y

        # Find how far particle can actually fall
        for step in prange(1, int(fall_distance + 1)):
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
