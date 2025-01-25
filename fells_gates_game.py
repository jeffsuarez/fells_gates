"""
Middlesex Fells Gate Running Game Simulator
This script simulates a gate running game in the Middlesex Fells Reservation.
"""

import networkx as nx
import random
from collections import defaultdict
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from datetime import datetime

@dataclass
class RunSimulation:
    """Class to store simulation results"""
    gate_order: List[int]
    total_distance: float
    max_gate_distance: float
    min_gate_distance: float
    avg_gate_distance: float
    total_time: float
    elevation_gain: float
    path_coordinates: List[Tuple[float, float]]

class FellsGateGame:
    """Main game class"""
    
    # Define map boundaries
    MAP_BOUNDS = {
        'lat_min': 42.4265,
        'lat_max': 42.4505,
        'lon_min': -71.1108,
        'lon_max': -71.0825
    }
    
    def __init__(self):
        """Initialize the game"""
        self.G = nx.Graph()
        self._setup_trail_network()
        self.start_point = (42.4183, -71.1067)  # Medford High School
        
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 3959.87433  # Earth's radius in miles

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def create_route_visualization(self, sim: RunSimulation, filename: Optional[str] = None) -> str:
        """Create a visualization of the route"""
        plt.figure(figsize=(15, 15))
        
        # Add water features
        landmarks = [
            (-71.1089, 42.4468, 0.015, 0.012, 'North Reservoir'),
            (-71.1062, 42.4385, 0.015, 0.012, 'Middle Reservoir'),
            (-71.1042, 42.4345, 0.015, 0.012, 'South Reservoir'),
            (-71.0925, 42.4428, 0.020, 0.015, 'Spot Pond'),
            (-71.0892, 42.4458, 0.008, 0.008, 'Doleful Pond'),
            (-71.0975, 42.4408, 0.008, 0.008, 'Quarter Mile Pond')
        ]

        for coord in landmarks:
            plt.gca().add_patch(Rectangle((coord[0], coord[1]), coord[2], coord[3], 
                                        facecolor='lightblue', alpha=0.3))
            plt.annotate(coord[4], (coord[0] + coord[2]/2, coord[1] + coord[3]/2),
                        ha='center', va='center', fontsize=8, alpha=0.7)
        
        # Plot all gates
        for gate_num, (coords, _) in self.gate_data.items():
            lat, lon = coords
            plt.plot(lon, lat, 'ro', markersize=8)
            plt.annotate(str(gate_num), (lon, lat), 
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Plot the route
        path_lons = [coord[1] for coord in sim.path_coordinates]
        path_lats = [coord[0] for coord in sim.path_coordinates]
        plt.plot(path_lons, path_lats, 'b-', linewidth=2, alpha=0.7)
        
        # Mark start/end point
        plt.plot(self.start_point[1], self.start_point[0], 'go', 
                markersize=12, label='Start/End')
        
        # Set the plot boundaries
        plt.xlim(self.MAP_BOUNDS['lon_min'], self.MAP_BOUNDS['lon_max'])
        plt.ylim(self.MAP_BOUNDS['lat_min'], self.MAP_BOUNDS['lat_max'])
        
        # Add title and labels
        plt.title('Middlesex Fells Gate Running Route\n' +
                 f'Total Distance: {sim.total_distance:.2f} miles, ' +
                 f'Time: {sim.total_time:.2f} hours')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.legend()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fells_route_{timestamp}.png'
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

    def get_path_details(self, gate1: int, gate2: int) -> Dict:
        """Calculate detailed path information between two gates"""
        try:
            path = nx.shortest_path(self.G, gate1, gate2, weight='weight')
            distance = 0
            elevation_gain = 0
            trail_names = []
            
            for i in range(len(path)-1):
                edge_data = self.G[path[i]][path[i+1]]
                distance += edge_data['weight']
                if 'trail' in edge_data:
                    trail_names.append(edge_data['trail'])
                if 'elevation_change' in edge_data:
                    elevation_gain += edge_data['elevation_change']
                    
            return {
                'path': path,
                'distance': distance,
                'elevation_gain': elevation_gain,
                'trails': list(set(trail_names))
            }
        except nx.NetworkXNoPath:
            return None

    def simulate_run(self, running_speed_mph: float, gate_stop_time_min: float, 
                    create_visualization: bool = False) -> RunSimulation:
        """Simulate a single gate running session"""
        available_gates = list(range(1, 59))
        current_position = self.start_point
        gate_order = []
        distances = []
        total_distance = 0
        total_elevation_gain = 0
        path_coordinates = [self.start_point]
        
        while available_gates:
            next_gate = random.choice(available_gates)
            available_gates.remove(next_gate)
            gate_order.append(next_gate)
            
            if len(gate_order) > 1:
                path_details = self.get_path_details(gate_order[-2], next_gate)
                if path_details:
                    distances.append(path_details['distance'])
                    total_distance += path_details['distance']
                    total_elevation_gain += path_details['elevation_gain']
                    for node in path_details['path']:
                        path_coordinates.append(self.gate_data[node][0])
                        
        # Calculate return to start
        if gate_order:
            start_node = min(self.G.nodes(), key=lambda n: 
                self._calculate_distance(self.start_point, self.gate_data[n][0]))
            path_details = self.get_path_details(gate_order[-1], start_node)
            if path_details:
                total_distance += path_details['distance']
                total_elevation_gain += path_details['elevation_gain']
                for node in path_details['path']:
                    path_coordinates.append(self.gate_data[node][0])
                path_coordinates.append(self.start_point)
            
        # Calculate statistics
        max_distance = max(distances) if distances else 0
        min_distance = min(distances) if distances else 0
        avg_distance = statistics.mean(distances) if distances else 0
        
        # Calculate total time
        elevation_time_hours = (total_elevation_gain / 100) / 60
        running_time_hours = total_distance / running_speed_mph
        gate_time_hours = (len(gate_order) * gate_stop_time_min) / 60
        total_time = running_time_hours + gate_time_hours + elevation_time_hours
        
        sim = RunSimulation(
            gate_order=gate_order,
            total_distance=total_distance,
            max_gate_distance=max_distance,
            min_gate_distance=min_distance,
            avg_gate_distance=avg_distance,
            total_time=total_time,
            elevation_gain=total_elevation_gain,
            path_coordinates=path_coordinates
        )
        
        if create_visualization:
            self.create_route_visualization(sim)
            
        return sim

    def _setup_trail_network(self):
        """Initialize the trail network with gates and trails"""
        # Gate locations (lat, lon) and elevations (feet)
        self.gate_data = {
            # Northern Fells Gates
            1: ((42.4468, -71.1089), 180),  # North Reservoir
            2: ((42.4455, -71.1078), 185),  # North Reservoir East
            3: ((42.4442, -71.1095), 175),  # North Reservoir West
            4: ((42.4425, -71.1082), 190),  # Between North and Middle
            5: ((42.4412, -71.1068), 195),  # Middle Reservoir North
            6: ((42.4398, -71.1075), 185),  # Middle Reservoir West
            7: ((42.4385, -71.1062), 180),  # Middle Reservoir South
            8: ((42.4372, -71.1048), 175),  # Between Middle and South
            9: ((42.4358, -71.1055), 170),  # South Reservoir North
            10: ((42.4345, -71.1042), 165),  # South Reservoir East
            
            # Add remaining gates with similar pattern
            # ... (Include all gates from 11-58 as previously defined)
            
            58: ((42.4265, -71.0858), 135)    # Final Gate
        }
        
        # Add gates as nodes
        for gate_num, (coords, elevation) in self.gate_data.items():
            self.G.add_node(gate_num, pos=coords, elevation=elevation)
            
        # Define trail connections (similar to before)
        trail_connections = [
            # Add all trail connections as previously defined
        ]
        
        # Add edges with properties
        for gate1, gate2, properties in trail_connections:
            if 'distance' not in properties:
                coord1 = self.gate_data[gate1][0]
                coord2 = self.gate_data[gate2][0]
                properties['distance'] = self._calculate_distance(coord1, coord2)
            
            elev1 = self.gate_data[gate1][1]
            elev2 = self.gate_data[gate2][1]
            properties['elevation_change'] = abs(elev2 - elev1)
            
            adjusted_distance = properties['distance'] * properties.get('difficulty', 1.0)
            elevation_penalty = properties['elevation_change'] * 0.001
            
            self.G.add_edge(gate1, gate2, 
                          **properties,
                          weight=adjusted_distance + elevation_penalty)

def run_simulation(num_simulations: int, running_speed_mph: float, gate_stop_time_min: float, 
                  visualize_best: bool = False, visualize_worst: bool = False):
    """Run multiple simulations and analyze the results"""
    game = FellsGateGame()
    simulations = []
    
    print(f"\nRunning {num_simulations} simulations...")
    start_time = time.time()
    
    for i in range(num_simulations):
        sim = game.simulate_run(running_speed_mph, gate_stop_time_min)
        simulations.append(sim)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} simulations...")
            
    end_time = time.time()
    
    # Analysis
    total_distances = [sim.total_distance for sim in simulations]
    total_times = [sim.total_time for sim in simulations]
    elevation_gains = [sim.elevation_gain for sim in simulations]
    
    # Find fastest and slowest routes
    fastest_sim = min(simulations, key=lambda x: x.total_time)
    slowest_sim = max(simulations, key=lambda x: x.total_time)
    
    if visualize_best:
        game.create_route_visualization(fastest_sim, 'fastest_route.png')
    if visualize_worst:
        game.create_route_visualization(slowest_sim, 'slowest_route.png')
    
    # Print results
    print("\nSimulation Results:")
    print("-" * 50)
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    print(f"\nAverage Statistics:")
    print(f"Average total distance: {statistics.mean(total_distances):.2f} miles")
    print(f"Average total time: {statistics.mean(total_times):.2f} hours")
    print(f"Average elevation gain: {statistics.mean(elevation_gains):.0f} feet")
    print(f"\nFastest Route:")
    print(f"Distance: {fastest_sim.total_distance:.2f} miles")
    print(f"Time: {fastest_sim.total_time:.2f} hours")
    print(f"Elevation gain: {fastest_sim.elevation_gain:.0f} feet")
    print(f"Gate order: {fastest_sim.gate_order}")
    print(f"\nSlowest Route:")
    print(f"Distance: {slowest_sim.total_distance:.2f} miles")
    print(f"Time: {slowest_sim.total_time:.2f} hours")
    print(f"Elevation gain: {fastest_sim.elevation_gain:.0f} feet")
    print(f"Gate order: {fastest_sim.gate_order}")
    print(f"\nSlowest Route:")
    print(f"Distance: {slowest_sim.total_distance:.2f} miles")
    print(f"Time: {slowest_sim.total_time:.2f} hours")
    print(f"Elevation gain: {slowest_sim.elevation_gain:.0f} feet")
    print(f"Gate order: {slowest_sim.gate_order}")

def get_valid_int_input(prompt: str, default: int) -> int:
    """Get valid integer input with a default value."""
    while True:
        try:
            user_input = input(f"{prompt} (default={default}): ").strip()
            if user_input == '':
                return default
            value = int(user_input)
            if value <= 0:
                print("Please enter a positive number.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

def get_valid_float_input(prompt: str, default: float) -> float:
    """Get valid float input with a default value."""
    while True:
        try:
            user_input = input(f"{prompt} (default={default}): ").strip()
            if user_input == '':
                return default
            value = float(user_input)
            if value <= 0:
                print("Please enter a positive number.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    # Default values
    DEFAULT_SIMS = 10
    DEFAULT_SPEED = 6.0
    DEFAULT_GATE_TIME = 1.0

    # Get user inputs with validation
    num_sims = get_valid_int_input("Enter number of simulations to run", DEFAULT_SIMS)
    running_speed = get_valid_float_input("Enter running speed (mph)", DEFAULT_SPEED)
    gate_time = get_valid_float_input("Enter time spent at each gate (minutes)", DEFAULT_GATE_TIME)
    
    # Get visualization preference with default
    visualize = input("Visualize routes? (y/n) [default=n]: ").lower().strip()
    visualize = visualize == 'y'
    
    # Run simulations
    run_simulation(num_sims, running_speed, gate_time, 
                  visualize_best=visualize, visualize_worst=visualize)

