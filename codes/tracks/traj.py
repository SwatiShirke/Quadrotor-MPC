import numpy as np
import pandas as pd

def circle_track(radius, num_points):
    """Generate a circular track."""
    theta = np.linspace(0, np.pi *2, num_points)  # Quarter circle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Calculate track length (s), yaw, and kappa
    s = np.zeros(num_points)
    for i in range(1, num_points):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    kappa = np.gradient(yaw) / np.gradient(s)
    
    return np.column_stack([s, x, y, yaw, kappa])


def sine_track(amplitude, frequency, num_points):
    """Generate a sine wave track."""
    x = np.linspace(0, 50, num_points)
    y = amplitude * np.sin(frequency * x)
    
    # Calculate track length (s), yaw, and kappa
    s = np.zeros(num_points)
    for i in range(1, num_points):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    kappa = np.gradient(yaw) / np.gradient(s)
    
    return np.column_stack([s, x, y, yaw, kappa])


def figure8_track(amplitude, num_points):
    """Generate a figure-eight track."""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = amplitude * np.sin(t)
    y = amplitude * np.sin(2 * t)  # Figure-eight
    
    # Calculate track length (s), yaw, and kappa
    s = np.zeros(num_points)
    for i in range(1, num_points):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    kappa = np.gradient(yaw) / np.gradient(s)
    
    return np.column_stack([s, x, y, yaw, kappa])


def spiral_track(radius_start, radius_end, num_points):
    """Generate a spiral track."""
    t = np.linspace(0, 4 * np.pi, num_points)
    radius = np.linspace(radius_start, radius_end, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    
    # Calculate track length (s), yaw, and kappa
    s = np.zeros(num_points)
    for i in range(1, num_points):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    kappa = np.gradient(yaw) / np.gradient(s)
    
    return np.column_stack([s, x, y, yaw, kappa])


def random_track(num_points):
    """Generate a random trajectory."""
    x = np.linspace(0, 10, num_points)
    y = np.random.randn(num_points) * 5  # Random vertical offsets
    
    # Calculate track length (s), yaw, and kappa
    s = np.zeros(num_points)
    for i in range(1, num_points):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    kappa = np.gradient(yaw) / np.gradient(s)
    
    return np.column_stack([s, x, y, yaw, kappa])


def generate_track_data(track_type, **kwargs):
    """Main function to call specific track generator functions."""
    if track_type == 'circle':
        return circle_track(**kwargs)
    elif track_type == 'sine':
        return sine_track(**kwargs)
    elif track_type == 'figure8':
        return figure8_track(**kwargs)
    elif track_type == 'spiral':
        return spiral_track(**kwargs)
    elif track_type == 'random':
        return random_track(**kwargs)
    else:
        raise ValueError(f"Unknown track type: {track_type}")

## Circle track with radius 10 and 100 points
circle_data = generate_track_data('circle', radius=20, num_points=100)
np.savetxt("tracks/circle.txt", circle_data, fmt="%.7e", delimiter="   ", header="", comments='')

# # Sine wave track with amplitude 5, frequency 1, and 100 points
sine_data = generate_track_data('sine', amplitude=10, frequency=0.15, num_points=100)
np.savetxt("tracks/sine.txt", sine_data, fmt="%.7e", delimiter="   ", header="", comments='')

# Figure-eight track with amplitude 5 and 100 points
figure8_data = generate_track_data('figure8', amplitude=5, num_points=100)
np.savetxt("tracks/fig8.txt", figure8_data, fmt="%.7e", delimiter="   ", header="", comments='')

# Spiral track with starting radius 5, ending radius 20, and 100 points
spiral_data = generate_track_data('spiral', radius_start=5, radius_end=20, num_points=100)
np.savetxt("tracks/spiral.txt", spiral_data, fmt="%.7e", delimiter="   ", header="", comments='')

# Random track with 100 points
random_data = generate_track_data('random', num_points=100)
np.savetxt("tracks/random.txt", random_data, fmt="%.7e", delimiter="   ", header="", comments='')
