import numpy as np
from typing import Dict, List, Tuple
from data_structures import User, Node, Link, obj
import networkx as nx
import logging

class ResourceAllocator:
    """
    Algorithm for resource allocation in the privacy-preserving federated learning network.
    This class implements a greedy approach for user-BS association, RB allocation,
    and server assignment based on fake locations.
    """
    def __init__(self, users: Dict[int, User], nodes: Dict[int, Node], 
                 links: Dict[int, Link], num_RBs: int, num_MCS: int, 
                 channel_gains: np.ndarray, nrTBSMatrix: np.ndarray,
                 noise_power: float, gamma_th: List[float], config: dict):
        """
        Initialize the resource allocator.
        
        Args:
            users (Dict[int, User]): Dictionary of User objects.
            nodes (Dict[int, Node]): Dictionary of Node objects.
            links (Dict[int, Link]): Dictionary of Link objects.
            num_RBs (int): Number of resource blocks.
            num_MCS (int): Number of modulation and coding schemes.
            channel_gains (np.ndarray): Channel gains between users and nodes.
            nrTBSMatrix (np.ndarray): TBS matrix for throughput calculation.
            noise_power (float): Noise power in linear scale.
            gamma_th (List[float]): SINR thresholds for each MCS level.
            config (dict): Configuration dictionary.
        """
        self.users = users
        self.nodes = nodes
        self.links = links
        self.num_RBs = num_RBs
        self.num_MCS = num_MCS
        self.channel_gains = channel_gains
        self.nrTBSMatrix = nrTBSMatrix
        self.noise_power = noise_power
        self.gamma_th = gamma_th
        self.config = config
        
        # Convert gamma_th from dB to linear scale
        self.gamma_th_linear = 10 ** (np.array(gamma_th) / 10)
        
        # Maximum transmission power in mW (linear scale)
        self.P_max = 10 ** (config['P_max_Tx_dBm'] / 10)
        
        # Build the graph for routing
        self.graph = self._build_graph()
        
        # Initialize resource trackers
        self._init_resources()
    
    def _build_graph(self):
        """
        Build a graph representation of the network for routing.
        
        Returns:
            nx.Graph: Network graph.
        """
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                       processing_capacity=node.processing_capacity,
                       used_capacity=0,
                       location=node.real_location,
                       node_type=node.node_type)
        
        # Add links with capacities as weights
        for link_id, link in self.links.items():
            node_1_id = link.node_1.id
            node_2_id = link.node_2.id
            # Use inverse capacity as weight (so shortest path favors high capacity)
            weight = 1.0 / link.link_capacity if link.link_capacity > 0 else float('inf')
            G.add_edge(node_1_id, node_2_id, 
                       capacity=link.link_capacity,
                       used_capacity=0,
                       weight=weight)
        
        return G
    
    def _init_resources(self):
        """Initialize resource trackers for BS, servers, and links."""
        # Resource blocks available per BS (assuming all nodes with type 'G' are BSs)
        self.available_RBs = {}
        # Processing capacity available per server
        self.available_processing = {}
        
        for node_id, node in self.nodes.items():
            if node.node_type == 'G':  # Ground stations are BSs
                self.available_RBs[node_id] = np.ones((self.num_RBs,), dtype=bool)  # True means available
            self.available_processing[node_id] = node.processing_capacity
        
        # Reset user assignments
        for user in self.users.values():
            user.associated_BS = None
            user.associated_server = None
            user.assigned_RBs = 0
            user.assigned_MCS = -1
            user.is_served = False
            user.achieved_throughput = 0
            user.achieved_delay = float('inf')
    
    def allocate(self):
        """
        Perform resource allocation for all users based on their fake locations.
        
        Returns:
            dict: Resource allocation results for metrics calculation.
        """
        # Reset resource tracking
        self._init_resources()
        
        # Get distances based on fake locations
        distances = obj.calculate_distance_between_groups(self.users, self.nodes, use_fake_distance=True)
        
        # Sort users by delay requirement (serve users with tighter constraints first)
        sorted_users = sorted(self.users.values(), key=lambda u: u.delay_requirement)
        
        # Keep track of allocations for constructing the solution
        solution = {
            'x': np.zeros((len(self.users), len(self.nodes), len(self.nodes))),  # Flow variables
            'y': np.zeros((len(self.users), len(self.nodes))),  # Server assignment
            'z': np.zeros((len(self.users), len(self.nodes))),  # BS assignment
            'zeta': np.zeros((len(self.users), self.num_RBs, self.num_MCS)),  # RB and MCS allocation
            'T_trans': np.zeros(len(self.users)),  # Transmission delay
            'T_proc': np.zeros(len(self.users)),  # Processing delay
            'lambda': np.zeros((len(self.users), len(self.nodes)))  # BS association
        }
        
        for user in sorted_users:
            # Try to allocate resources for this user
            result = self._allocate_user(user, distances[user.id])
            if result['served']:
                # Update solution matrix
                bs_id = result['bs_id']
                server_id = result['server_id']
                fake_bs_id = result['fake_bs_id']
                
                # BS assignment
                solution['z'][user.id, bs_id] = 1
                
                # Server assignment
                solution['y'][user.id, server_id] = 1
                
                # Link between real BS and fake BS
                if bs_id != fake_bs_id:
                    solution['x'][user.id, bs_id, fake_bs_id] = 1
                
                # Link between fake BS and server
                if fake_bs_id != server_id:
                    solution['x'][user.id, fake_bs_id, server_id] = 1
                
                # RB and MCS allocation
                rb_count = result['rb_count']
                mcs = result['mcs']
                
                # Mark the first rb_count RBs as assigned with the selected MCS
                if rb_count > 0 and mcs >= 0:
                    # Find allocated RBs
                    allocated_rbs = np.where(result['allocated_rbs'])[0][:rb_count]
                    for rb in allocated_rbs:
                        solution['zeta'][user.id, rb, mcs] = 1
                
                # BS association (lambda)
                solution['lambda'][user.id, bs_id] = 1
                
                # Delay components
                solution['T_trans'][user.id] = result['transmission_delay']
                solution['T_proc'][user.id] = result['processing_delay']
                
                logging.info(f"User {user.id} served: BS {bs_id}, Fake BS {fake_bs_id}, Server {server_id}, "
                          f"RBs {rb_count}, MCS {mcs}, Throughput {result['throughput']:.2f}, "
                          f"Delay {result['total_delay']:.4f}/{user.delay_requirement:.4f}")
            else:
                logging.info(f"User {user.id} NOT served. Reason: {result['reason']}")
        
        return solution
    
    def _allocate_user(self, user, distances):
        """
        Allocate resources for a single user.
        
        Args:
            user (User): User to allocate resources for.
            distances (np.ndarray): Distances from user to all nodes.
            
        Returns:
            dict: Resource allocation result for the user.
        """
        # Find closest base station using fake location
        bs_candidates = self._find_bs_candidates(user, distances)
        
        if not bs_candidates:
            return {'served': False, 'reason': 'No BS available'}
        
        # For each BS candidate, find the best MCS and required RBs
        best_allocation = None
        
        for bs_id, distance in bs_candidates:
            # Skip if no RBs available at this BS
            if not np.any(self.available_RBs[bs_id]):
                continue
            
            # Calculate SINR and select MCS
            sinr = self._calculate_sinr(user.id, bs_id)
            mcs = self._select_mcs(sinr)
            
            if mcs == -1:
                continue  # SINR too low for any MCS
            
            # Calculate required RBs for the user's rate requirement
            required_rbs, throughput = self._calculate_required_rbs(mcs, user.rate_requirement)
            
            if required_rbs == -1:
                continue  # Cannot satisfy rate requirement
              # Check if we have enough RBs available at this BS
            available_mask = self.available_RBs[bs_id]
            if np.sum(available_mask) < required_rbs:
                continue  # Not enough RBs
            
            # Calculate transmission delay with protection against division by zero
            if throughput <= 0:
                logging.warning(f"Invalid throughput {throughput} for user {user.id}. Skipping allocation.")
                continue  # Skip this allocation
            transmission_delay = (1 / throughput) * user.rate_requirement
            
            # Find server with routing through fake BS
            server_result = self._find_server_and_route(user, bs_id)
            
            if not server_result['success']:
                continue  # No valid server and route found
            
            # Calculate total delay
            total_delay = transmission_delay + server_result['processing_delay']
            
            # Check if delay requirement is satisfied
            if total_delay > user.delay_requirement:
                continue  # Delay requirement not satisfied
            
            # We found a valid allocation, keep track of the best one (minimizing delay)
            if best_allocation is None or total_delay < best_allocation['total_delay']:                # Allocate the required RBs
                allocated_rbs = np.where(available_mask)[0][:required_rbs]
                
                best_allocation = {
                    'served': True,
                    'bs_id': bs_id,
                    'fake_bs_id': server_result['fake_bs_id'],
                    'server_id': server_result['server_id'],
                    'mcs': mcs,
                    'rb_count': required_rbs,
                    'allocated_rbs': allocated_rbs.copy(),
                    'transmission_delay': transmission_delay,
                    'processing_delay': server_result['processing_delay'],
                    'total_delay': total_delay,
                    'throughput': throughput,
                    'path': server_result['path']
                }
        
        if best_allocation is None:
            return {'served': False, 'reason': 'No valid resource allocation found'}
        
        # Commit the best allocation
        bs_id = best_allocation['bs_id']
        server_id = best_allocation['server_id']
        required_rbs = best_allocation['rb_count']
        allocated_rbs = np.where(best_allocation['allocated_rbs'])[0][:required_rbs]
        
        # Mark RBs as used
        for rb in allocated_rbs:
            self.available_RBs[bs_id][rb] = False
        
        # Update server processing capacity
        self.available_processing[server_id] -= user.rate_requirement
        
        # Reduce link capacities along the path
        path = best_allocation['path']
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.graph[u][v]['used_capacity'] += user.rate_requirement
        
        # Update user properties
        user.set_user_properties(
            associated_BS=bs_id,
            associated_server=server_id,
            assigned_RBs=required_rbs,
            assigned_MCS=mcs,
            is_served=True,
            achieved_throughput=best_allocation['throughput'],
            achieved_delay=best_allocation['total_delay']
        )
        
        return best_allocation
    
    def _find_bs_candidates(self, user, distances):
        """
        Find base station candidates for a user, sorted by distance.
        
        Args:
            user (User): User to find BS candidates for.
            distances (np.ndarray): Distances from user to all nodes.
            
        Returns:
            list: List of (bs_id, distance) tuples sorted by distance.
        """
        bs_candidates = []
        
        for node_id, node in self.nodes.items():
            if node.node_type == 'G':  # Ground stations are BSs
                bs_candidates.append((node_id, distances[node_id]))
          # Sort by distance
        bs_candidates.sort(key=lambda x: x[1])
        
        return bs_candidates
    
    def _calculate_sinr(self, user_id, bs_id):
        """
        Calculate SINR for a user-BS pair.
        
        Args:
            user_id (int): User ID.
            bs_id (int): BS ID.
            
        Returns:
            float: SINR value in linear scale.
        """
        # Use the average channel gain across all RBs for simplicity
        # In a real system, you would use the specific RB's channel gain
        channel_gain = np.mean(self.channel_gains[user_id, bs_id, :])
        
        # Calculate receive power
        rx_power = self.P_max * channel_gain
        
        # Calculate SINR with protection against division by zero
        if self.noise_power <= 0:
            logging.warning(f"Invalid noise power: {self.noise_power}. Using default value.")
            noise_power = 1e-12  # Very small but non-zero value
        else:
            noise_power = self.noise_power
        
        sinr = rx_power / noise_power
        
        return sinr
    
    def _select_mcs(self, sinr):
        """
        Select the highest MCS that the SINR can support.
        
        Args:
            sinr (float): SINR value in linear scale.
            
        Returns:
            int: Selected MCS index, or -1 if none is suitable.
        """
        # Find the highest MCS where SINR >= threshold
        for mcs in range(self.num_MCS - 1, -1, -1):
            if sinr >= self.gamma_th_linear[mcs]:
                return mcs
        return -1  # No suitable MCS found
    
    def _calculate_required_rbs(self, mcs, rate_requirement):
        """
        Calculate the minimum number of RBs needed to satisfy the rate requirement.
        
        Args:
            mcs (int): MCS index.
            rate_requirement (float): Rate requirement in Mbps.
            
        Returns:
            tuple: (required_rbs, achievable_throughput), where required_rbs is -1 if not possible.
        """
        # For each number of RBs, check if it can satisfy the rate requirement
        for rb_count in range(1, self.num_RBs + 1):
            # Use the TBS matrix to calculate the throughput
            # Handle the nrTBSMatrix whether it's a numpy array or a dictionary
            if isinstance(self.nrTBSMatrix, dict):
                # If it's a dictionary with tuple keys (rb_count, mcs)
                throughput = self.nrTBSMatrix.get((rb_count, mcs), 0)
            else:
                # If it's a numpy array with [rb_count-1, mcs] indexing
                throughput = self.nrTBSMatrix[rb_count - 1, mcs]
            
            if throughput >= rate_requirement:
                return rb_count, throughput
        
        return -1, 0  # Cannot satisfy the rate requirement
    
    def _find_server_and_route(self, user, bs_id):
        """
        Find a server and route from the BS through a fake BS.
        
        Args:
            user (User): User object.
            bs_id (int): Real BS ID.
            
        Returns:
            dict: Server and routing result.
        """
        # Find the closest BS to the fake location (fake BS)
        fake_distances = []
        for node_id, node in self.nodes.items():
            if node.node_type == 'G':  # Ground stations are BSs
                fake_distance = obj.calculate_distance(user, node, use_fake_distance=True)
                fake_distances.append((node_id, fake_distance))
        
        # Sort by distance to fake location
        fake_distances.sort(key=lambda x: x[1])
        
        # Try each potential fake BS
        for fake_bs_id, _ in fake_distances:
            # Find servers with enough processing capacity
            server_candidates = []
            for node_id, capacity in self.available_processing.items():
                if capacity >= user.rate_requirement:
                    server_candidates.append(node_id)
            
            if not server_candidates:
                return {'success': False, 'reason': 'No server with enough capacity'}
            
            # For each server, check if we can route from bs -> fake_bs -> server
            for server_id in server_candidates:
                try:
                    # First check if we can route from real BS to fake BS
                    if bs_id != fake_bs_id:
                        path1 = self._check_path(bs_id, fake_bs_id, user.rate_requirement)
                        if not path1:
                            continue
                    else:
                        path1 = [bs_id]
                    
                    # Then check if we can route from fake BS to server
                    if fake_bs_id != server_id:
                        path2 = self._check_path(fake_bs_id, server_id, user.rate_requirement)
                        if not path2:
                            continue
                    else:
                        path2 = [fake_bs_id]
                    
                    # Combine paths (remove duplicate fake_bs_id)
                    full_path = path1 + path2[1:] if len(path2) > 1 else path1
                      # Calculate processing delay based on server's capacity
                    # Protect against division by zero
                    if self.nodes[server_id].processing_capacity <= 0:
                        logging.warning(f"Server {server_id} has zero or negative processing capacity. Using default delay.")
                        processing_delay = 10.0  # Default high delay for servers with no capacity
                    else:
                        # Assuming processing delay is inversely proportional to available capacity
                        processing_delay = 1.0 / self.nodes[server_id].processing_capacity
                    
                    return {
                        'success': True,
                        'fake_bs_id': fake_bs_id,
                        'server_id': server_id,
                        'path': full_path,
                        'processing_delay': processing_delay
                    }
                
                except nx.NetworkXNoPath:
                    continue  # No path found, try the next server
        
        return {'success': False, 'reason': 'No valid route found'}
    
    def _check_path(self, source, target, required_capacity):
        """
        Check if there's a path from source to target with enough capacity.
        
        Args:
            source (int): Source node ID.
            target (int): Target node ID.
            required_capacity (float): Required capacity in Mbps.
            
        Returns:
            list: Path nodes if valid, None otherwise.
        """
        # Try to find the shortest path
        try:
            path = nx.shortest_path(self.graph, source=source, target=target, weight='weight')
        except nx.NetworkXNoPath:
            return None
        
        # Check capacity along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.graph[u][v]
            if edge['capacity'] - edge['used_capacity'] < required_capacity:
                return None  # Not enough capacity
        
        return path
    
    def allocate_resources_heuristic(self):
        """
        Allocate resources using the heuristic approach.
        This is a wrapper around the allocate method to maintain compatibility.
        
        Returns:
            dict: Resource allocation results for metrics calculation.
        """
        # Run the allocation algorithm
        solution = self.allocate()
        
        # Calculate additional metrics for the solution
        served_users = sum(1 for user in self.users.values() if user.is_served)
        total_users = len(self.users)
        served_percentage = (served_users / total_users) * 100 if total_users > 0 else 0
        
        avg_throughput = sum(user.achieved_throughput for user in self.users.values()) / total_users if total_users > 0 else 0
        avg_delay = sum(user.achieved_delay for user in self.users.values() if user.is_served) / served_users if served_users > 0 else float('inf')
        
        metrics = {
            'served_percentage': served_percentage,
            'avg_throughput': avg_throughput,
            'avg_delay': avg_delay,
            'solution': solution
        }
        
        return metrics
