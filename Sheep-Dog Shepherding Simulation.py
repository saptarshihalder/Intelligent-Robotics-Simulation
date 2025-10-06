import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

class Sheep:
    """Individual sheep agent with flocking behavior"""
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.random.randn(2) * 0.5
        self.radius = 0.3
        
    def update(self, sheep_list, dog, target, params, bounds):
        """Update sheep position based on flocking rules and dog repulsion"""
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        neighbor_count = 0
        
        # Flocking behavior with other sheep
        for other in sheep_list:
            if other is self:
                continue
                
            diff = other.pos - self.pos
            dist = np.linalg.norm(diff)
            
            # Cohesion: move towards average position of neighbors
            if dist < params['attraction_radius']:
                cohesion += other.pos
                neighbor_count += 1
            
            # Separation: avoid crowding
            if 0 < dist < params['repulsion_radius']:
                separation -= diff / dist
        
        # Apply cohesion
        if neighbor_count > 0:
            cohesion = cohesion / neighbor_count - self.pos
            self.vel += cohesion * params['cohesion_weight']
        
        # Apply separation
        self.vel += separation * params['separation_weight']
        
        # Dog repulsion - flee from dog
        dog_diff = self.pos - dog.pos
        dog_dist = np.linalg.norm(dog_diff)
        
        if dog_dist < params['dog_influence_radius'] and dog_dist > 0:
            repulsion_strength = params['dog_repulsion_weight'] * \
                                (1 - dog_dist / params['dog_influence_radius'])
            self.vel += (dog_diff / dog_dist) * repulsion_strength
        
        # Limit speed
        speed = np.linalg.norm(self.vel)
        if speed > params['sheep_max_speed']:
            self.vel = (self.vel / speed) * params['sheep_max_speed']
        
        # Update position
        self.pos += self.vel
        
        # Boundary conditions - bounce off walls
        for i in range(2):
            if self.pos[i] < bounds[i][0]:
                self.pos[i] = bounds[i][0]
                self.vel[i] *= -0.5
            elif self.pos[i] > bounds[i][1]:
                self.pos[i] = bounds[i][1]
                self.vel[i] *= -0.5


class Dog:
    """Herding dog agent that drives sheep towards target"""
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.zeros(2)
        self.radius = 0.4
        
    def update(self, sheep_list, target, params):
        """Update dog position using shepherding strategy"""
        if not sheep_list:
            return
        
        # Calculate center of mass of sheep
        sheep_center = np.mean([sheep.pos for sheep in sheep_list], axis=0)
        
        # Find furthest sheep from target
        furthest_sheep = max(sheep_list, 
                            key=lambda s: np.linalg.norm(s.pos - target))
        
        # Strategy: position behind the furthest sheep relative to target
        to_target = target - furthest_sheep.pos
        target_dist = np.linalg.norm(to_target)
        
        if target_dist > 0.1:
            # Position dog behind the furthest sheep
            behind_pos = furthest_sheep.pos - (to_target / target_dist) * 2.0
            
            # Move towards optimal position
            desired_vel = behind_pos - self.pos
            dist = np.linalg.norm(desired_vel)
            
            if dist > 0:
                desired_vel = (desired_vel / dist) * params['dog_max_speed']
                self.vel = desired_vel
        else:
            # If sheep are at target, move to center
            desired_vel = sheep_center - self.pos
            dist = np.linalg.norm(desired_vel)
            if dist > 0:
                self.vel = (desired_vel / dist) * params['dog_max_speed'] * 0.5
        
        # Update position
        self.pos += self.vel


class ShepherdingSimulation:
    """Main simulation class"""
    def __init__(self, num_sheep=20, bounds=((-10, 10), (-10, 10))):
        self.bounds = bounds
        
        # Simulation parameters
        self.params = {
            'repulsion_radius': 1.5,
            'attraction_radius': 3.0,
            'dog_influence_radius': 4.0,
            'cohesion_weight': 0.03,
            'separation_weight': 0.15,
            'dog_repulsion_weight': 0.8,
            'sheep_max_speed': 0.3,
            'dog_max_speed': 0.5
        }
        
        # Initialize sheep randomly in one area
        self.sheep = [
            Sheep(
                np.random.uniform(-8, -4),
                np.random.uniform(-8, 8)
            ) for _ in range(num_sheep)
        ]
        
        # Initialize dog on opposite side
        self.dog = Dog(8, 0)
        
        # Target location (where we want to herd sheep)
        self.target = np.array([8.0, 0.0])
        
    def update(self):
        """Update one simulation step"""
        # Update all sheep
        for sheep in self.sheep:
            sheep.update(self.sheep, self.dog, self.target, 
                        self.params, self.bounds)
        
        # Update dog
        self.dog.update(self.sheep, self.target, self.params)
    
    def animate(self, frames=500, interval=50):
        """Create animation of the simulation"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Sheep-Dog Shepherding Simulation', fontsize=14, fontweight='bold')
        
        # Plot elements
        sheep_scatter = ax.scatter([], [], c='white', s=200, edgecolors='black', 
                                  linewidths=2, label='Sheep', zorder=3)
        dog_scatter = ax.scatter([], [], c='brown', s=300, marker='^', 
                                edgecolors='black', linewidths=2, label='Dog', zorder=4)
        target_scatter = ax.scatter([self.target[0]], [self.target[1]], 
                                   c='green', s=500, marker='*', 
                                   edgecolors='darkgreen', linewidths=2, 
                                   label='Target', zorder=2, alpha=0.6)
        
        # Dog influence radius (visual aid)
        dog_circle = Circle(self.dog.pos, self.params['dog_influence_radius'], 
                          fill=False, linestyle='--', color='brown', alpha=0.3)
        ax.add_patch(dog_circle)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right')
        
        def init():
            sheep_scatter.set_offsets(np.empty((0, 2)))
            dog_scatter.set_offsets(np.empty((0, 2)))
            return sheep_scatter, dog_scatter, dog_circle, time_text
        
        def animate_frame(frame):
            # Update simulation
            self.update()
            
            # Update sheep positions
            sheep_pos = np.array([sheep.pos for sheep in self.sheep])
            sheep_scatter.set_offsets(sheep_pos)
            
            # Update dog position
            dog_scatter.set_offsets([self.dog.pos])
            
            # Update dog influence circle
            dog_circle.center = self.dog.pos
            
            # Update time
            time_text.set_text(f'Time: {frame}')
            
            return sheep_scatter, dog_scatter, dog_circle, time_text
        
        anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                      frames=frames, interval=interval, 
                                      blit=True, repeat=True)
        
        plt.tight_layout()
        return anim


# Run simulation
if __name__ == "__main__":
    print("Starting Sheep-Dog Shepherding Simulation...")
    print("=" * 50)
    
    # Create simulation
    sim = ShepherdingSimulation(num_sheep=20)
    
    print(f"Number of sheep: {len(sim.sheep)}")
    print(f"Target location: {sim.target}")
    print(f"Dog starting position: {sim.dog.pos}")
    print("\nSimulation Parameters:")
    for key, value in sim.params.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Create and display animation
    anim = sim.animate(frames=500, interval=50)
    
    # To save animation (optional, requires ffmpeg or pillow)
    # anim.save('shepherding_simulation.gif', writer='pillow', fps=20)
    
    plt.show()
    print("\nSimulation complete!")
