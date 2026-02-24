# Particle Filter for Robot Localization
# Inputs: 
#   - sensors dict (landmarks map, observations, odometry)
#   - previous particles and weights
#   - filter type indicator

def particle_filter_step(particles, weights, sensors, control):
    """
    One step of the particle filter algorithm.
    Returns updated particles, weights, and best pose estimate.
    """
    
    # --- CONSTANTS (you will need to tune these) ---
    NUM_PARTICLES = 500
    SENSOR_RANGE = 5.0  # meters, maximum sensor range
    ALPHA = [0.1, 0.1, 0.05, 0.05]  # motion noise parameters
    SIGMA_RANGE = 0.1  # range measurement noise (meters)
    SIGMA_BEARING = 0.05  # bearing measurement noise (radians)
    RANDOM_PARTICLE_RATIO = 0.05  # for kidnapped robot recovery
    
    # Extract data from sensors dict
    landmarks = sensors['map']  # list of landmark IDs and positions
    observations = sensors['observations']  # list of (range, bearing, landmark_id)
    odometry = sensors['odometry']  # (dx, dy, dtheta) since last step
    
    # --- STEP 1: Initialize (if first time step) ---
    if particles is None:
        # Global localization: spread particles uniformly across arena
        particles = initialize_particles_uniform(NUM_PARTICLES, ARENA_SIZE)
        weights = [1.0 / NUM_PARTICLES] * NUM_PARTICLES
    
    # --- STEP 2: Prediction (Motion Update) ---
    new_particles = []
    for i in range(NUM_PARTICLES):
        # Apply motion model with noise
        particle = particles[i]
        
        # Add noise to odometry (simulate motion uncertainty)
        noisy_odometry = add_motion_noise(odometry, ALPHA, particle)
        
        # Move particle according to noisy odometry
        new_particle = apply_motion(particle, noisy_odometry)
        new_particles.append(new_particle)
    
    # --- STEP 3: Correction (Measurement Update) ---
    new_weights = []
    for i in range(NUM_PARTICLES):
        particle = new_particles[i]
        
        # Calculate weight based on how well particle matches observations
        weight = 1.0
        
        if observations is not None and len(observations) > 0:
            # For each observation
            for obs in observations:
                obs_range = obs['range']
                obs_bearing = obs['bearing']
                obs_landmark_id = obs['landmark_id']
                
                # Find this landmark in the map
                landmark = get_landmark_by_id(landmarks, obs_landmark_id)
                
                # Predict what sensor should see from this particle
                expected_range, expected_bearing = predict_measurement(particle, landmark)
                
                # Skip if landmark is out of sensor range (shouldn't be seen)
                if expected_range > SENSOR_RANGE:
                    continue
                
                # Calculate likelihood of this observation
                # Assuming Gaussian noise model
                range_prob = gaussian_probability(obs_range, expected_range, SIGMA_RANGE)
                bearing_prob = gaussian_probability(obs_bearing, expected_bearing, SIGMA_BEARING)
                
                # Multiply probabilities (assuming independence)
                obs_likelihood = range_prob * bearing_prob
                
                # Multiply weight by observation likelihood
                weight *= obs_likelihood
        
        # Store weight
        new_weights.append(weight)
    
    # --- STEP 4: Normalize Weights ---
    total_weight = sum(new_weights)
    if total_weight > 0:
        new_weights = [w / total_weight for w in new_weights]
    else:
        # If all weights are zero, reinitialize (catastrophic failure)
        new_particles = initialize_particles_uniform(NUM_PARTICLES, ARENA_SIZE)
        new_weights = [1.0 / NUM_PARTICLES] * NUM_PARTICLES
    
    # --- STEP 5: Resampling ---
    resampled_particles = []
    resampled_weights = []
    
    # Systematic resampling (low variance resampling)
    positions = (np.random.random() + np.arange(NUM_PARTICLES)) / NUM_PARTICLES
    cumulative_sum = np.cumsum(new_weights)
    i = 0
    for pos in positions:
        while cumulative_sum[i] < pos:
            i += 1
        resampled_particles.append(new_particles[i].copy())
        resampled_weights.append(1.0 / NUM_PARTICLES)  # Equal weights after resampling
    
    # --- STEP 6: Inject Random Particles (for kidnapped robot recovery) ---
    num_random = int(RANDOM_PARTICLE_RATIO * NUM_PARTICLES)
    if num_random > 0:
        for j in range(num_random):
            # Replace some particles with random ones
            idx = np.random.randint(NUM_PARTICLES)
            resampled_particles[idx] = create_random_particle(ARENA_SIZE)
            # Weight stays uniform
    
    # --- STEP 7: Estimate Best Pose ---
    # Use weighted average of particles (or find the cluster)
    best_pose = estimate_pose_from_particles(resampled_particles, resampled_weights)
    
    # --- STEP 8: Generate Control Command ---
    # Simple exploration: move toward unexplored areas or just random walk
    control = generate_exploration_control(best_pose, landmarks)
    
    return resampled_particles, resampled_weights, best_pose, control


# --- Helper Functions ---

def initialize_particles_uniform(num_particles, arena_size):
    """Initialize particles uniformly across the arena."""
    particles = []
    for _ in range(num_particles):
        x = np.random.uniform(-arena_size/2, arena_size/2)
        y = np.random.uniform(-arena_size/2, arena_size/2)
        theta = np.random.uniform(-np.pi, np.pi)
        particles.append([x, y, theta])
    return particles

def add_motion_noise(odometry, alpha, particle):
    """Add noise to odometry based on motion model."""
    dx, dy, dtheta = odometry
    
    # Simple noise model: noise proportional to movement
    # More sophisticated models use alpha parameters
    noise_x = np.random.normal(0, alpha[0] * abs(dx) + alpha[1] * abs(dtheta))
    noise_y = np.random.normal(0, alpha[0] * abs(dy) + alpha[1] * abs(dtheta))
    noise_theta = np.random.normal(0, alpha[2] * abs(dtheta) + alpha[3] * (abs(dx) + abs(dy)))
    
    return [dx + noise_x, dy + noise_y, dtheta + noise_theta]

def apply_motion(particle, noisy_odometry):
    """Apply odometry to particle pose."""
    x, y, theta = particle
    dx, dy, dtheta = noisy_odometry
    
    # Update pose
    new_x = x + dx * np.cos(theta) - dy * np.sin(theta)
    new_y = y + dx * np.sin(theta) + dy * np.cos(theta)
    new_theta = theta + dtheta
    
    # Normalize angle to [-pi, pi]
    new_theta = ((new_theta + np.pi) % (2 * np.pi)) - np.pi
    
    return [new_x, new_y, new_theta]

def predict_measurement(particle, landmark):
    """Predict range and bearing to landmark from particle pose."""
    x, y, theta = particle
    lx, ly = landmark['x'], landmark['y']
    
    # Vector from robot to landmark
    dx = lx - x
    dy = ly - y
    
    # Range
    range_pred = np.sqrt(dx**2 + dy**2)
    
    # Bearing (angle to landmark in robot's local frame)
    bearing_pred = np.arctan2(dy, dx) - theta
    bearing_pred = ((bearing_pred + np.pi) % (2 * np.pi)) - np.pi
    
    return range_pred, bearing_pred

def gaussian_probability(value, mean, std_dev):
    """Compute probability of value given Gaussian distribution."""
    if std_dev <= 0:
        return 1.0
    exponent = -0.5 * ((value - mean) / std_dev) ** 2
    return (1.0 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)

def estimate_pose_from_particles(particles, weights):
    """Estimate best pose from particle set."""
    # Simple weighted average
    x = sum(p[0] * w for p, w in zip(particles, weights))
    y = sum(p[1] * w for p, w in zip(particles, weights))
    
    # Average angle carefully (circular mean)
    sin_sum = sum(np.sin(p[2]) * w for p, w in zip(particles, weights))
    cos_sum = sum(np.cos(p[2]) * w for p, w in zip(particles, weights))
    theta = np.arctan2(sin_sum, cos_sum)
    
    return [x, y, theta]

def generate_exploration_control(best_pose, landmarks):
    """Generate control commands for exploration."""
    # Simple strategy: move forward and turn slightly
    # This could be enhanced with active localization
    
    # If no landmarks seen recently, turn more to find them
    # Otherwise, move forward
    
    # Basic control: move forward at 0.5 m/s, turn slowly
    v = 0.5  # linear velocity
    omega = 0.1  # angular velocity
    
    return {'v': v, 'omega': omega}

def get_landmark_by_id(landmarks, landmark_id):
    """Retrieve landmark from map by ID."""
    for landmark in landmarks:
        if landmark['id'] == landmark_id:
            return landmark
    return None