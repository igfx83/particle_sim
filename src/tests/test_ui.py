# Ultra-minimal particle for testing
class MinimalParticle:
    __slots__ = ["element_id", "x", "y", "temp"]

    def __init__(self, element_id, x=0, y=0):
        self.element_id = element_id
        self.x = x
        self.y = y
        self.temp = 20.0


# Test with minimal particles
def test_minimal_particles(simulation_grid):
    """Replace your particles with minimal ones temporarily"""

    # Clear existing particles
    simulation_grid.particles.clear()

    # Add 50k minimal particles
    start = time.perf_counter()
    for i in range(50000):
        particle = MinimalParticle("water", i % 200, i // 200)
        simulation_grid.particles.append(particle)
    creation_time = time.perf_counter() - start

    print(f"Minimal particle creation: {creation_time*1000:.2f}ms")

    # Test iteration
    def minimal_update(dt):
        start = time.perf_counter()
        count = 0
        for particle in simulation_grid.particles:
            if particle is not None:
                count += 1
                # Minimal processing
                particle.temp += 0.001
        end = time.perf_counter()
        print(f"Minimal update: {(end-start)*1000:.2f}ms for {count} particles")

    # Run test
    for _ in range(5):
        minimal_update(1 / 60)


# Alternative: Pure array approach (should be fastest)
class ArrayParticleSystem:
    def __init__(self, max_particles=100000):
        # All particle data as numpy arrays
        self.element_ids = np.zeros(max_particles, dtype=np.uint16)
        self.x_coords = np.zeros(max_particles, dtype=np.uint16)
        self.y_coords = np.zeros(max_particles, dtype=np.uint16)
        self.temperatures = np.full(max_particles, 20.0, dtype=np.float32)
        self.active = np.zeros(max_particles, dtype=bool)

        self.count = 0

    def add_particle(self, element_id, x, y):
        if self.count >= len(self.element_ids):
            return False

        idx = self.count
        self.element_ids[idx] = hash(element_id) % 65536  # Simple hash
        self.x_coords[idx] = x
        self.y_coords[idx] = y
        self.active[idx] = True
        self.count += 1
        return True

    def update(self, dt):
        start = time.perf_counter()

        # Vectorized operations on all active particles
        active_mask = self.active[: self.count]
        active_count = np.sum(active_mask)

        if active_count > 0:
            # Example: Heat all particles slightly (vectorized)
            self.temperatures[: self.count][active_mask] += 0.001

            # Example: Simple gravity (vectorized)
            # self.y_coords[:self.count][active_mask] -= 1

        end = time.perf_counter()
        print(f"Array update: {(end-start)*1000:.2f}ms for {active_count} particles")


# Test the array approach
def test_array_system():
    system = ArrayParticleSystem()

    # Add 50k particles
    start = time.perf_counter()
    for i in range(50000):
        system.add_particle("water", i % 200, i // 200)
    creation_time = time.perf_counter() - start
    print(f"Array creation: {creation_time*1000:.2f}ms")

    # Test updates
    for _ in range(10):
        system.update(1 / 60)


# Memory-efficient particle storage
class CompactParticle:
    """Particle that stores data more efficiently"""

    __slots__ = ["_data"]  # Single slot for packed data

    def __init__(self, element_id, x=0, y=0):
        # Pack multiple values into a single integer/bytes object
        # This is extreme optimization - probably overkill
        element_hash = hash(element_id) & 0xFFFF  # 16 bits
        x_val = x & 0xFFFF  # 16 bits
        y_val = y & 0xFFFF  # 16 bits
        temp_val = int(20.0 * 100) & 0xFFFF  # 16 bits (temp * 100)

        # Pack into 64-bit integer
        self._data = (element_hash << 48) | (x_val << 32) | (y_val << 16) | temp_val

    @property
    def x(self):
        return (self._data >> 32) & 0xFFFF

    @property
    def y(self):
        return (self._data >> 16) & 0xFFFF

    @property
    def temperature(self):
        return (self._data & 0xFFFF) / 100.0

    @temperature.setter
    def temperature(self, value):
        temp_bits = int(value * 100) & 0xFFFF
        self._data = (self._data & 0xFFFFFFFFFFFF0000) | temp_bits


# Quick benchmark function
def benchmark_approaches(count=50000):
    """Compare different particle storage approaches"""

    approaches = {
        "List[MinimalParticle]": lambda: [
            MinimalParticle("water", i % 200, i // 200) for i in range(count)
        ],
        "List[CompactParticle]": lambda: [
            CompactParticle("water", i % 200, i // 200) for i in range(count)
        ],
        "ArrayParticleSystem": lambda: ArrayParticleSystem(),  # Special case
    }

    for name, creator in approaches.items():
        if name == "ArrayParticleSystem":
            start = time.perf_counter()
            system = creator()
            for i in range(count):
                system.add_particle("water", i % 200, i // 200)
            creation_time = time.perf_counter() - start
        else:
            start = time.perf_counter()
            particles = creator()
            creation_time = time.perf_counter() - start

        print(f"{name} creation: {creation_time*1000:.2f}ms for {count} particles")

        # Test iteration
        if name == "ArrayParticleSystem":
            for _ in range(3):
                system.update(1 / 60)
        else:
            for _ in range(3):
                start = time.perf_counter()
                processed = 0
                for p in particles:
                    if p is not None:
                        processed += 1
                        # Minimal processing
                        if hasattr(p, "temp"):
                            p.temp += 0.001
                end = time.perf_counter()
                print(
                    f"{name} iteration: {(end-start)*1000:.2f}ms for {processed} particles"
                )
