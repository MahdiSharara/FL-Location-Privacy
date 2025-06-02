import numpy as np
from typing import Dict, Tuple
from utilities.load_json import load_config
config = load_config('config.json')
x_y_location_range = config['x_y_location_range']
class obj:
    def __init__(self, id: int, real_location: np.ndarray = None):
        """
        Initialize a base object.

        Args:
            id (int): Object ID.
            real_location (np.ndarray, optional): Real location of the object (x, y, z).
        """
        self.id = id
        if real_location is not None:
            self.real_location = real_location  # Optional for objects like Link

    @staticmethod
    def calculate_distance(obj1, obj2, use_fake_distance) -> float:
        """
        Calculate the Euclidean distance between two objects based on their locations.
        """
        location1 = obj1.fake_location if (use_fake_distance and hasattr(obj1, 'fake_location') and obj1.fake_location is not None) else obj1.real_location
        location2 = obj2.fake_location if (use_fake_distance and hasattr(obj2, 'fake_location') and obj2.fake_location is not None) else obj2.real_location

        # Ensure both locations are valid
        if location1 is None or location2 is None:
            raise ValueError(f"Invalid location: location1={location1}, location2={location2}, obj1={obj1}, obj2={obj2}")
        #round to the the nearest 2 decimal point
        rounded_distance = np.linalg.norm(location1 - location2)
        rounded_distance = np.round(rounded_distance, 2)

        return rounded_distance

    @staticmethod
    def calculate_distance_between_groups(group_1, group_2, use_fake_distance=False):
        """
        Calculate the distance matrix between two groups of objects.

        Args:
            group_1: Dictionary of objects in the first group (e.g., Users).
            group_2: Dictionary of objects in the second group (e.g., Nodes).
            use_fake_distance: Boolean indicating whether to use fake location.

        Returns:
            np.ndarray: Distance matrix between the two groups.
        """
        num_group_1 = len(group_1)
        num_group_2 = len(group_2)
        d_real = np.zeros((num_group_1, num_group_2))  # Correctly initialize the array

        for i, (item_1_id, item_1) in enumerate(group_1.items()):
            for j, (item_2_id, item_2) in enumerate(group_2.items()):
                d_real[i, j] = item_1.calculate_distance(item_2, use_fake_distance)
        rounded_d_real = np.round(d_real, 2)  # Round the distances to 2 decimal places
        return rounded_d_real
        
class User(obj):
    def __init__(self, id: int, rate_requirement: float, delay_requirement: float, 
                 max_epsilon: float, real_location: np.ndarray = None):
        """
        Initialize a User object.

        Args:
            id (int): User ID.
            real_location (np.ndarray): Real location of the user (x, y, z).
            rate_requirement (float): Rate requirement of the user.
            delay_requirement (float): Delay requirement of the user.
            max_epsilon (float): Maximum epsilon value for differential privacy.
        """
        super().__init__(id, real_location)
        self.rate_requirement = rate_requirement
        self.delay_requirement = delay_requirement
        self.max_epsilon = max_epsilon

        # Ensure real_location is properly initialized
        if self.real_location is None:
            raise ValueError(f"User {id} must have a valid real_location.")

        self.assigned_epsilon = None
        self.fake_location = None
        self.associated_BS = None
        self.associated_server = None
        self.assigned_RBs = None
        self.assigned_MCS = None
        self.is_served = False
        self.achieved_throughput = None
        self.achieved_delay = None

    def set_epsilon_generate_fake_location(self, epsilon: float):
        """
        Set the epsilon value for differential privacy and generate the fake location.

        Args:
            epsilon (float): Epsilon value for differential privacy.
        """
        self.assigned_epsilon = epsilon
        self.fake_location = self.generate_fake_location()

    def generate_fake_location(self) -> np.ndarray:
        """
        Generate a fake location using the Laplace mechanism.

        Returns:
            np.ndarray: Fake location.
        """
        if self.assigned_epsilon is None:
            raise ValueError("Epsilon value must be set before generating fake location.")
        noise = np.random.laplace(0, config['deltaF']/self.assigned_epsilon, size=self.real_location.shape[0]-1)
        fake_location = self.real_location + np.concatenate((noise , [0]))
        #if x and y values of the fake location are out of the defined range in config variable, we have to set them to the limit
        # of the range.
        
        new_location = np.clip(fake_location, x_y_location_range[0], x_y_location_range[1])
        
        return new_location

    def calculate_distance(self, obj2, fake_distance):
        """
        Calculate the distance between this user and another object.

        Args:
            obj2: The second object (User or Node).
            fake_distance: Boolean indicating whether to use fake location.

        Returns:
            float: The Euclidean distance between the two objects.
        """
        return obj.calculate_distance(self, obj2, fake_distance)

    def set_user_properties(self, associated_BS=None, associated_server=None, 
                             assigned_RBs=None, assigned_MCS=None, 
                             is_served=None, achieved_throughput=None, 
                             achieved_delay=None):
        """
        Set additional properties for the User object.

        Args:
            associated_BS: Associated base station.
            associated_server: Associated server.
            assigned_RBs: Assigned resource blocks.
            assigned_MCS: Assigned modulation and coding scheme.
            is_served (bool): Whether the user is served.
            achieved_throughput: Achieved throughput.
            achieved_delay: Achieved delay.
        """
        if associated_BS is not None:
            self.associated_BS = associated_BS
        if associated_server is not None:
            self.associated_server = associated_server
        if assigned_RBs is not None:
            self.assigned_RBs = assigned_RBs
        if assigned_MCS is not None:
            self.assigned_MCS = assigned_MCS
        if is_served is not None:
            self.is_served = is_served
        if achieved_throughput is not None:
            self.achieved_throughput = achieved_throughput
        if achieved_delay is not None:
            self.achieved_delay = achieved_delay
            
class Node(obj):
    def __init__(self, id: int, real_location: np.ndarray, 
                 processing_capacity: float, node_type: str ):
        """
        Initialize a Node object.

        Args:
            id (int): Node ID.
            real_location (np.ndarray): Real location of the node (x, y, z).
            processing_capacity (float): Processing capacity of the node.
        """
        super().__init__(id, real_location)  # Call the obj constructor
        self.processing_capacity = processing_capacity
        self.node_type = node_type  # 'G' or 'D' Type of node (e.g., Ground, Drone, etc.)
    def calculate_distance(self, obj2, fake_distance=False):
        """
        Calculate the distance between this node and another object.

        Args:
            obj2: The second object (User or Node).
            fake_distance: Boolean indicating whether to use fake location.

        Returns:
            float: The Euclidean distance between the two objects.
        """
        return obj.calculate_distance(self, obj2, fake_distance)
# In the following there is no source and destination. since it is bidirectional
# we should have two nodes that link connect to each other
class Link(obj):
    # we should have two nodes that link connect to each other
    # if a link connects node_1 to node_2 then it is connecting node_2 to node_1
    def __init__(self, id: int, node_1: Node, node_2: Node,
                 link_capacity: float):
        """
        Initialize a Link object.

        Args:
            id (int): Link ID.
            node_1 (Node): First node object. if a link connects node_1 to node_2 then it is connecting node_2 to node_1
            node_2 (Node): Second node object. if a link connects node_1 to node_2 then it is connecting node_2 to node_1
            capacity (float): Capacity of the link.
        """
        super().__init__(id)  # Call the obj constructor (no real_location for Link)
        self.node_1 = node_1
        self.node_2 = node_2
        self.link_capacity = link_capacity
        self.delay = 1/link_capacity