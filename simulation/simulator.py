import sumolib
import libsumo
import traci
import os
import sys
import random
import torch

class SUMOTrafficSimulator:
    """
    Class responsible for initializing, running, managing, and terminating the SUMO traffic simulation.
    """
    def __init__(self, configs, logger=None):
        self.sumo_configs = configs
        self.logger = logger
        self.metrics = self.sumo_configs['metrics']
        self.simulation_config = self.sumo_configs['simulation']
        self.interface_type = self.simulation_config['interface_type']

    def initialize_sumo(self):
        """
        Initialize the SUMO simulation with proper configurations and error handling.
        """
        self.logger.info("Initializing SUMO simulation.")

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            self.logger.error("SUMO_HOME environment variable not set.")
            sys.exit("Please set the 'SUMO_HOME' environment variable.")

        self._load_simulation_config()
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.gui else 'sumo')
        self.sumo_cmd = [sumo_binary, '-c', self.sumocfg_file]
        if self.no_warning:
            self.sumo_cmd.append('--no-warnings')

        # Add collision options
        self.sumo_cmd += [
            '--collision.action', 'remove',  # Remove vehicles upon collision
            '--collision.check-junctions',   # Check for collisions at junctions
            '--collision.mingap-factor', '0' # Only physical collisions are registered
        ]

        self.connect_to_sumo()
        self.initialize_elements()
        self.network_graph = self.build_graph()


    def connect_to_sumo(self):
        """
        Connect to the SUMO simulation using the specified interface (libsumo or traci).
        """
        if self.interface_type == "libsumo":
            libsumo.start(self.sumo_cmd)
            self.sumo_interface = libsumo
        elif self.interface_type == "traci":
            traci.start(self.sumo_cmd)
            self.sumo_interface = traci
        else:
            raise ValueError("Invalid interface type specified.")
        self.simulation_running = True
        self.is_truncated = False
        self.is_terminated = False
        self.simulation_max_steps = int(self.sumo_interface.simulation.getEndTime())
        self.simulation_step = int(self.sumo_interface.simulation.getTime())
        self.vehicle_data = {}  # Dictionary to store vehicle data

    def close_sumo(self):
        """
        Close the SUMO simulation.
        """
        if self.simulation_running:
            self.logger.info("Closing SUMO simulation.") 
                       
            if self.interface_type == "libsumo":
                libsumo.close()
            elif self.interface_type == "traci":
                traci.close()
            self.simulation_running = False
            self.is_terminated = True

    def _load_simulation_config(self):
        """
        Load initial configuration for the simulation, including traffic lights and detectors.
        """
        self.simulation_seed = self.simulation_config['seed']
        self.gui = self.simulation_config['gui']
        self.no_warning = self.simulation_config['no_warnings']
        self.network_name = self.simulation_config['network_name']
        self.interface_type = self.simulation_config['interface_type']
        self.sumocfg_file = os.path.join(
            self.simulation_config['networks_path'],
            self.network_name,
            self.network_name + "_sumo_config.sumocfg"
        )
        random.seed(self.simulation_seed)

        # Load accident configuration
        self.accident_interval = self.simulation_config.get('accident_interval', None)  # e.g., every 100 steps
        self.accident_probability = self.simulation_config.get('accident_probability', 0.0)  # e.g., 0.01 for 1% chance
        self.accident_duration = self.simulation_config.get('accident_duration', 30)  # e.g., 30 seconds

    def simulate_accident(self):
        """
        Simulate an accident by manipulating vehicle parameters to cause a collision.
        """
        # Get all vehicles
        vehicle_ids = self.sumo_interface.vehicle.getIDList()
        if not vehicle_ids:
            self.logger.info("No vehicles in simulation to cause accident")
            return

        # Try to find a pair of vehicles on the same lane
        for lane_id in self.sumo_interface.lane.getIDList():
            vehicles_on_lane = self.sumo_interface.lane.getLastStepVehicleIDs(lane_id)
            if len(vehicles_on_lane) >= 2:
                # Get the first two vehicles (front and back)
                front_vehicle = vehicles_on_lane[0]
                back_vehicle = vehicles_on_lane[1]

                # Set the front vehicle to stop suddenly
                self.sumo_interface.vehicle.setSpeed(front_vehicle, 0)
                self.sumo_interface.vehicle.setDecel(front_vehicle, 9.0)  # Maximum deceleration

                # Disable safety checks for the back vehicle
                self.sumo_interface.vehicle.setSpeedMode(back_vehicle, 0)
                # Increase the speed of the back vehicle
                current_speed = self.sumo_interface.vehicle.getSpeed(back_vehicle)
                self.sumo_interface.vehicle.setSpeed(back_vehicle, current_speed + 10)

                self.logger.info(f"Simulating accident between vehicles {front_vehicle} and {back_vehicle} on lane {lane_id}")

                return

        self.logger.info("No suitable vehicles found on same lane to simulate accident")

    def apply_road_restrictions(self):
        """
        Simulate road restrictions by blocking lanes or edges in the network.
        """
        # Fetch road restrictions
        mapped_events = self.road_restrictions_fetcher.get_filtered_events()
        lane_impact_mapping = {
            'ALL LANES CLOSED': [],
            '1 Right Lane(s)': '_0',
            '2 Right Lane(s)': ['_0', '_1'],
        }

        # Apply road restrictions
        for event in mapped_events:
            edge_id = event['edge_id']
            edge = self.sumo_interface.edge.getEdge(edge_id)
            if event['event']['LanesAffected'] == 'ALL LANES CLOSED':
                edge.setAllowed('')  # Close all lanes
            else:
                # Only close specific lane (rightmost) edge_ID_0
                for lane in edge.getLanes():
                    if lane.getID().endswith(lane_impact_mapping[event['event']['LanesAffected']]):
                        lane.setAllowed('')
                        break

        construction_data = self.road_restrictions_fetcher.get_construction_projects(mapped_events)
        
        for construction in construction_data:
            edge_id = construction['edge_id']
            edge = self.sumo_interface.edge.getEdge(edge_id)
            if construction['is_full_closure']:
                edge.setAllowed('')  # Close all lanes
            else:
                # Only close specific lane (rightmost) edge_ID_0
                for lane in edge.getLanes():
                    if lane.getID().endswith(lane_impact_mapping[event['event']['LanesAffected']]):
                        lane.setAllowed('')
                        break
        

    def initialize_elements(self):
        """
        Initialize the traffic light controllers for the simulation.
        """

        self.lanes = self.sumo_interface.lane.getIDList()
        self.edges = self.sumo_interface.edge.getIDList()
        self.traffic_light_ids = self.sumo_interface.trafficlight.getIDList()


    def build_graph(self):
        """
        Construct the graph based on the interface type (libsumo or traci).
        This graph will be used for Graph Neural Network operations.
        """
        if self.logger:
            self.logger.info("Building the traffic network graph using TraCI.")

        # Initialize node mapping: traffic light IDs to sequential indices
        node_idx_map = {tls_id: idx for idx, tls_id in enumerate(self.traffic_light_ids)}
        num_nodes = len(self.traffic_light_ids)
        # Prepare lists to collect edge indices
        edge_index = [[], []]  # [source_nodes, target_nodes]

        # Set of traffic light nodes for quick lookup
        traffic_light_nodes = set(self.traffic_light_ids)

        # Build the graph by iterating over each traffic light controller
        for tls_id in self.traffic_light_ids:
            src_idx = node_idx_map[tls_id]
            controlled_lanes = self.sumo_interface.trafficlight.getControlledLanes(tls_id)

            for lane_id in controlled_lanes:
                # Get the outgoing connections from the current lane
                links = self.sumo_interface.lane.getLinks(lane_id)
                for link in links:
                    outgoing_lane_id = link[0]
                    # Get the edge ID associated with the outgoing lane
                    edge_id = self.sumo_interface.lane.getEdgeID(outgoing_lane_id)
                    # Get the destination node (junction) of the edge
                    to_node_id = self.sumo_interface.edge.getToJunction(edge_id)
                    # Check if the destination node is controlled by a traffic light
                    if to_node_id in traffic_light_nodes:
                        dest_idx = node_idx_map[to_node_id]
                        # Add an edge from the current node to the destination node
                        edge_index[0].append(src_idx)
                        edge_index[1].append(dest_idx)

        # Convert edge indices to a PyTorch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Store the graph attributes for later use
        self.graph = {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'node_idx_map': node_idx_map,
            'idx_node_map': {idx: tls_id for tls_id, idx in node_idx_map.items()}
        }

        if self.logger:
            self.logger.info("Traffic network graph construction completed.")

        return self.graph

    def advance_simulation_step(self):
        """
        Apply actions and advance the simulation state.
        Check for termination conditions: maximum steps reached or no more vehicles.
        """
        try:
            # Check for maximum steps reached
            if self.simulation_step >= self.simulation_max_steps:
                self.logger.info("Simulation step limit reached. Ending simulation.")
                self.is_truncated = True
                return self.is_truncated, self.is_terminated

            # Check if all vehicles have arrived
            elif self.sumo_interface.simulation.getMinExpectedNumber() == 0:
                self.logger.info("All vehicles have arrived at their destinations. Ending simulation.")
                self.is_terminated = True
                return self.is_truncated, self.is_terminated

            # # Trigger accidents based on interval or probability
            # if self.accident_interval and self.simulation_step % self.accident_interval == 0:
            #     self.simulate_accident()
            # elif self.accident_probability > 0.0 and random.random() < self.accident_probability:
            #     self.simulate_accident()
                
            # # Check for road restrictions every 10 steps
            # if self.simulation_step % 10 == 0:
            #     self.apply_road_restrictions()


            self.sumo_interface.simulationStep()
            self.simulation_step += 1
            self.collect_vehicle_data()

            return self.is_truncated, self.is_terminated

        except Exception as e:
            self.logger.error(f"Error updating SUMO state: {e}")
            raise


    def collect_vehicle_data(self):
        """
        Collect data about vehicles at each simulation step.
        """
        current_time = self.simulation_step
        vehicle_ids = self.sumo_interface.vehicle.getIDList()

        # Track new vehicles entering the network
        for veh_id in vehicle_ids:
            if veh_id not in self.vehicle_data:
                # Vehicle has just entered the network
                self.vehicle_data[veh_id] = {
                    'entry_time': current_time,
                    'exit_time': None,
                    'waiting_time': 0.0,
                }
            else:
                # Update waiting time
                self.vehicle_data[veh_id]['waiting_time'] += self.sumo_interface.vehicle.getWaitingTime(veh_id)

        # Update exit times for vehicles that have left the network
        arrived_vehicle_ids = self.sumo_interface.simulation.getArrivedIDList()
        for veh_id in arrived_vehicle_ids:
            if veh_id in self.vehicle_data and self.vehicle_data[veh_id]['exit_time'] is None:
                self.vehicle_data[veh_id]['exit_time'] = current_time

    def get_vehicle_data(self):
        """
        Return the collected vehicle data.
        """
        return self.vehicle_data
    

    def reset_sumo(self):
        """
        Reset the simulation to initial conditions.
        """
        self.logger.info("Resetting SUMO simulation.")
        self.close_sumo()
        if not self.simulation_running:
            self.connect_to_sumo()

