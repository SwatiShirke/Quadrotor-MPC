from dataclasses import dataclass
import inspect
import yaml


@dataclass
class MPC:
    T_sim: float           # Simulation time
    Tf: float             # Final time (horizon time)
    N: int                # Number of steps (horizon length)
    m: float              # Mass of the robot
    g: float              # Gravitational constant (m/s^2)
    l: float              # Length (in meters)
    sigma: float          # Some coefficient
    ixx: float            # Moment of inertia around x-axis
    iyy: float            # Moment of inertia around y-axis
    izz: float            # Moment of inertia around z-axis
    r1: float             # Resistance 1 (e.g., friction, damping)
    r2: float             # Resistance 2
    r3: float             # Resistance 3
    n1: float             # Noise factor 1 (e.g., sensor noise, disturbances)
    n2: float             # Noise factor 2
    n3: float             # Noise factor 3
    N_sim: int            # Simulation steps
    u_max: float          # Maximum control input
    mav_name: str         # MAV name as a string
    mass: float           # Mass of the MAV
    Q: list[float]        # Weight on state costs
    R: list[float]        # Weight on input costs
    ubu: list[float]      # Upper bound on inputs
    lbu: list[float]      # Lower bound on inputs
    max_u: float
    ref_array: list


def dict_to_class(class_name, data):
    
    return class_name(
      **{
        key: data[key] if val.default == val.empty else data.get(key, val.default)
        for key, val in inspect.signature(class_name).parameters.items()
      }
    )


def read_yaml(yaml_path):
    #print(yaml_path)
    with open(yaml_path) as file:
        try:
            parsed_file = yaml.safe_load(file)
            print("pased yaml file")
            return parsed_file
        except:
            print("could not parsed the yaml file")
            return None