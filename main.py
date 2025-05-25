#Had an oopsie, lost the entire script cos github, had to get chatgpt to recite all my code back to me so it mught be a bit weird
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import interpolate
import copy
from tqdm import tqdm
from matplotlib import cm
from shapely.geometry import LineString
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import neuralfoil as nf
import multiprocessing
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
plt.rcParams["font.family"] = "Times New Roman"

class Trajectory(object):
    def __init__(self, surface, wire_trajectory, wire_velocity = []):
        self.surface_nodes = surface
        self.wire_trajectory_nodes = wire_trajectory
        self.wire_velocity = wire_velocity
        self.critical_surface_energy = 22

    def update(self,i):
        self.partial_exposure = np.sum(self.unsummed_surface_exposure[:, :i], axis=1)
        exposure_norm = self.partial_exposure / np.max(self.surface_exposure)
        colors = cm.jet(exposure_norm)
        self.exposure_lc.set_color(colors)
        self.trajectory_line.set_data(self.wire_trajectory_nodes[:i,0],self.wire_trajectory_nodes[:i,1])

    def predict_sde(self, verbose = False):
        self.surface_midpoints = np.array([(self.surface_nodes[n+1] + self.surface_nodes[n])/2 for n in range(self.surface_nodes.shape[0]-1)])
        self.surface_vectors = np.diff(self.surface_nodes,axis=0)
        self.surface_areas = [np.hypot(x,y) for x,y in  self.surface_vectors]
        self.wire_midpoints = np.array([(self.wire_trajectory_nodes[n+1] + self.wire_trajectory_nodes[n])/2 for n in range(self.wire_trajectory_nodes.shape[0]-1)])
        self.wire_vectors = np.diff(self.wire_trajectory_nodes,axis=0)
        self.wire_areas = [np.hypot(x,y) for x,y in  self.wire_vectors]
        self.surface_irradance = np.zeros_like(self.surface_nodes)

        self.diff = self.wire_midpoints[:, np.newaxis, :] - self.surface_midpoints[np.newaxis, :, :]
        self.distances = np.linalg.norm(self.diff, axis=2)

        self.cross_products = (
            self.surface_vectors[np.newaxis, :, 0] * self.diff[:, :, 1] -
            self.surface_vectors[np.newaxis, :, 1] * self.diff[:, :, 0]
        )
        self.visibility_matrix = (0 >= self.cross_products.T)
        self.dot_products = np.abs(
            self.surface_vectors[np.newaxis, :, 0] * self.wire_vectors[:, 0][:, np.newaxis] +
            self.surface_vectors[np.newaxis, :, 1] * self.wire_vectors[:, 1][:, np.newaxis]
        )
        self.perspective_matrix = (self.dot_products.T * self.wire_areas * self.surface_areas * np.tri(*self.visibility_matrix.shape,k=1)) / (self.distances.T**2)
        self.surface_exposure = np.sum(self.visibility_matrix * self.perspective_matrix, axis=1)

        if verbose:
            plt.imshow(self.visibility_matrix, interpolation='spline16', cmap = "binary")
            plt.colorbar()
            plt.grid()
            plt.ylabel("Surface panel index")
            plt.xlabel("Wire panel index")
            plt.title("Wire Surface Visibility")
            plt.gca().invert_yaxis()
            plt.show()

            plt.imshow(self.perspective_matrix*self.visibility_matrix, interpolation='nearest')
            plt.gca().invert_yaxis()
            plt.ylabel("Surface panel index")
            plt.xlabel("Wire panel index")
            plt.colorbar()
            plt.show()

            self.fig, self.ax = plt.subplots(figsize = (8,3), dpi = 100)
            self.ax.axhline(22,-20,220,color = "Red")
            self.ax.plot(self.surface_exposure, c = 'black')
            self.ax.set_xlabel("Surface panel index")
            self.ax.set_ylabel("Dimensionless heating parameter")
            self.ax.grid()
            plt.tight_layout()
            plt.show()

            self.fig,self.axs = plt.subplots()
            for n in range(0,len(self.surface_nodes)-1):
                self.axs.plot(self.surface_nodes[n:n+2,0],self.surface_nodes[n:n+2,1],c=cm.jet(self.surface_exposure[n]/np.max(self.surface_exposure)))
            plt.gca().set_aspect('equal')
            plt.title("Surface heating")
            plt.xlabel("X position [mm]")
            plt.ylabel("Y position [mm]")
            plt.tight_layout()
            plt.grid()
            plt.show()

            self.surface_segments = np.array([[self.surface_nodes[i], self.surface_nodes[i + 1]] for i in range(len(self.surface_nodes) - 1)])
            self.exposure_lc = LineCollection(self.surface_segments, linewidths=2)

            self.fig, self.axs = plt.subplots()
            self.axs.add_collection(self.exposure_lc)
            self.axs.set_aspect('equal')
            self.axs.set_xlim(np.min(self.surface_nodes[:, 0]) - 5, np.max(self.surface_nodes[:, 0]) + 5)
            self.axs.set_ylim(np.min(self.surface_nodes[:, 1]) - 5, np.max(self.surface_nodes[:, 1]) + 5)
            self.axs.grid(True)
            self.trajectory_line, = self.axs.plot([],[], c = "Red")
            self.unsummed_surface_exposure  = self.visibility_matrix * self.perspective_matrix
            self.animation = FuncAnimation(self.fig, self.update, interval = len(self.surface_nodes)-1)
            plt.show()

        return self.surface_exposure
class Panel(object):
    def __init__(self, p1,p2):
        self.points = np.vstack((p1,p2))
        self.panel_vector = np.diff(self.points,axis=0)[0]
        self.area = np.hypot(self.panel_vector[0],self.panel_vector[1])
        self.mid_point = np.mean(self.points,axis=0)
        self.direction = np.sign(np.diff(self.points[:,0],axis = 0)[0])

    def update(self):
        self.mid_point = np.mean(self.points,axis=0)
        self.direction = np.sign(np.diff(self.points[:,0],axis = 0)[0])
        self.area = np.hypot(self.panel_vector[0],self.panel_vector[1])
        self.line()

class Shape(object):
    def __init__(self, root, tip, root_chord, tip_chord, root_alpha, tip_alpha, sweep_angle):
        self.root = root
        self.tip = tip
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.root_alpha = root_alpha
        self.tip_alpha = tip_alpha
        self.sweep = sweep_angle
        self.Compute_goemetry()
        self.Fit_bspline_surface()

    def Discretize(self, points):
        return np.array([Panel(points[0,n], points[0,n+1]) for n in range(0,points.shape[1]-1)])

    def Compute_goemetry(self):
        self.root_discrete_directrices = self.rotate_data(self.root*self.root_chord, self.root_alpha, np.array([self.root_chord*0.3,0]))
        self.tip_discrete_directrices = self.rotate_data(self.tip*self.tip_chord, self.tip_alpha, np.array([self.tip_chord*0.3,0]))

    def rotate_data(self,data,alpha,rot_axis):
        self.rotated_dat= []
        for self.point in data:
            self.rel_point = self.point-rot_axis
            self.p_point = [np.sqrt(self.rel_point[0]**2+self.rel_point[1]**2),np.arctan2(self.rel_point[1],self.rel_point[0])]
            self.p_point[1]-= np.radians([alpha])
            self.c_points = np.array([self.p_point[0]*np.cos(self.p_point[1]),self.p_point[0]*np.sin(self.p_point[1])]).flatten()
            self.rotated_dat.append(self.c_points+rot_axis)
        self.rotated_dat = np.array(self.rotated_dat)
        return self.rotated_dat

    def Fit_bspline_surface(self):
        self.r_tck, _ = interpolate.splprep([self.root_discrete_directrices[:,0],self.root_discrete_directrices[:,1]],k=5,s=0.1,per=True)
        self.t_tck, _ = interpolate.splprep([self.tip_discrete_directrices[:,0],self.tip_discrete_directrices[:,1]],k=5,s=0.1,per=True)

    def Tip_geometry(self):
        self.Samples = np.linspace(0,1,200, endpoint = False)
        return interpolate.splev(self.Samples,self.t_tck)

    def Root_geometry(self):
        self.Samples = np.linspace(0,1,100, endpoint = False)
        return interpolate.splev(self.Samples,self.r_tck)

class main(object):
    def __init__(self):
        '''Loading in foil data'''
        self.foil_addr = "Airfoils//S1223.dat"
        self.raw = open(self.foil_addr,'r').read()
        self.foil_dat = np.array(self.format_dat(self.raw))[2:-2]
        self.Particles = 10
        self.k = 3
        self.Smoothing = 0.001
        self.Max_iterations = 15000
        '''Global Optimisation Parameters'''
        self.C1 = 0.9 # Local
        self.C2 = 0.1 # Global
        self.w = 0.6  # Inertia

        self.mainloop()
        self.Export()

    def add_arrow_to_line2D(self, axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>', arrowsize=1, transform=None):
        if not isinstance(line, mlines.Line2D):
            raise ValueError("expected a matplotlib.lines.Line2D object")
        x, y = line.get_xdata(), line.get_ydata()

        arrow_kw = {
            "arrowstyle": arrowstyle,
            "mutation_scale": 10 * arrowsize,
        }

        color = line.get_color()
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            raise NotImplementedError("multicolor lines not supported")
        else:
            arrow_kw['color'] = color

        linewidth = line.get_linewidth()
        if isinstance(linewidth, np.ndarray):
            raise NotImplementedError("multiwidth lines not supported")
        else:
            arrow_kw['linewidth'] = linewidth

        if transform is None:
            transform = axes.transData

        arrows = []
        for loc in arrow_locs:
            s = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            n = np.searchsorted(s, s[-1] * loc)
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
            p = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform, **arrow_kw)
            axes.add_patch(p)
            arrows.append(p)
        return arrows

    def Discretize(self, points):
        points = np.concatenate([points[0],[points[0,0]]], axis=0)
        return np.array([[points[n], points[n+1]] for n in range(0, points.shape[0]-1)])

    def MSE_Loss(self, array, target):
        return np.mean((array - target)**2)

    def format_dat(self, data):
        self.data = data.split("\n")
        self.formatted = [self.el.split(" ") for self.el in self.data]
        self.formatted = [[float(self.num) for self.num in list(filter(lambda x: x != '', self.coord))] for self.coord in self.formatted]
        self.formatted = list(filter(lambda x: x != [], self.formatted))
        return self.formatted

    def Offset(self, nodes, offsets):
        self.nodes = np.concatenate([nodes[-1:], nodes, nodes[:1]], axis=0)
        self.derivatives = self.nodes[1:] - self.nodes[:-1]
        self.Unit_derivaitves = self.derivatives / np.sqrt(np.sum(self.derivatives**2, axis=1))[:, None]
        self.Offset_unit_vectors = self.Unit_derivaitves[:, ::-1]
        self.Offset_unit_vectors[:, 0] *= -1
        self.Offset_unit_vectors = (self.Offset_unit_vectors[1:] + self.Offset_unit_vectors[:-1]) / 2
        self.Offset_vectors = self.Offset_unit_vectors * offsets[:, None] * -1
        return nodes + self.Offset_vectors

    def Objective_function(self, BSpline, verbose=False):
        self.trajectory_nodes = interpolate.splev(np.linspace(0,1,200, endpoint=False), BSpline)
        self.Trajectory_line = LineString([(self.trajectory_nodes[0][i], self.trajectory_nodes[1][i]) for i,_ in enumerate(self.trajectory_nodes[0])])
        if not self.Tip_line.intersection(self.Trajectory_line).is_empty:
            return 1e15
        self.trajectory_nodes = np.dstack((self.trajectory_nodes[0], self.trajectory_nodes[1]))[0]
        self.trajectory = Trajectory(self.Tip_points, self.trajectory_nodes)
        self.surface_exposure = self.trajectory.predict_sde(verbose)
        self.Exposure_loss = np.mean((self.surface_exposure - self.trajectory.critical_surface_energy)**2)
        return self.Exposure_loss

    def mainloop(self):
        self.shape = Shape(self.foil_dat, self.foil_dat, 300, 200, 2, 0, 5)
        self.x, self.y = self.shape.Tip_geometry()
        self.Tip_points = np.dstack((self.x, self.y))[0]
        self.Tip_line = LineString([(self.Tip_points[i, 0], self.Tip_points[i, 1]) for i, _ in enumerate(self.Tip_points)])

        self.offset = 2
        self.prev_offset = 0
        self.previous_loss = 1000
        while True:
            print(self.offset, self.offset - self.prev_offset)
            self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:, 0], self.offset))
            self.offset_points[0] += [self.offset, 0]
            self.tck, _ = interpolate.splprep([self.offset_points[:, 0], self.offset_points[:, 1]], k=self.k, s=0.02, per=False)
            self.t, self.c, self.k = self.tck
            self.loss = self.Objective_function([self.t, self.c, self.k])
            if self.previous_loss < self.loss:
                break
            self.learning_rate_modifier = ((1 / (1 + np.e**(-self.loss / 1000))) - 0.5) * 2
            self.prev_offset = self.offset
            self.offset = self.offset - 0.1 * self.learning_rate_modifier * np.sign(self.previous_loss - self.loss)

        self.offset_points = self.Offset(self.Tip_points, np.full_like(self.Tip_points[:, 0], self.offset))
        self.offset_points[0] += [self.offset, 0]

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_title("Tip foil profile and inital wire trajectory")
        self.ax.set_ylabel("Y position [mm]")
        self.ax.set_xlabel("X position [mm]")
        self.ax.plot(self.Tip_points[:, 0], self.Tip_points[:, 1], color="black")
        self.line, = self.ax.plot(self.offset_points[:, 0], self.offset_points[:, 1], color="red")
        self.add_arrow_to_line2D(self.ax, self.line, arrow_locs=np.linspace(0., 1., 200), arrowstyle='->')
        plt.show()

        self.tck = interpolate.splprep([self.offset_points[:, 0], self.offset_points[:, 1]], k=self.k, s=self.Smoothing, per=False)[0]
        self.t, self.c, self.k = self.tck
        self.v = np.dstack((self.c[0], self.c[1]))[0]
        self.Objective_function(self.tck, True)

        self.Particle_Positions = np.array([self.v for _ in range(self.Particles)])
        self.Particle_Velocity = np.array([(np.random.randn(*self.v.shape) - 0.5) * 4 for _ in range(self.Particles)])
        self.Particle_best_positions = np.copy(self.Particle_Positions)
        self.Particle_performance = np.array([self.Objective_function([self.t, [particle[:, 0], particle[:, 1]], self.k]) for particle in self.Particle_Positions])
        self.Particle_best_performance = np.copy(self.Particle_performance)
        self.Global_best = np.argmin(self.Particle_performance)
        self.Global_best_position = np.copy(self.Particle_Positions[self.Global_best])
        self.Global_best_performance = self.Particle_performance[self.Global_best]

        self.fig, self.ax = plt.subplots()
        self.loss_array = []
        self.Last_global_best = 0
        self.no_change_count = 0
        self.best_index = np.argmin(self.Particle_performance)
        self.design_variables = []
        self.design_var_loss = []
        self.i = 0
        while True:
            self.i += 1
            if self.i == 100:
                print(self.i)
                self.i = 0
                if (self.loss_array[-99] - self.loss_array[-1]) < 0.05:
                    break

            print("Iteration: ", self.i, "Global best performance: ", self.Global_best_performance,
                  " Iteration best: ", self.Particle_performance[self.best_index])
            self.Particle_Positions += self.Particle_Velocity
            self.loss_array.append(self.Global_best_performance)
            self.data = [[self.t, [particle[:, 0], particle[:, 1]], self.k] for particle in self.Particle_Positions]
            self.Particle_performance = np.array([self.Objective_function(var) for var in self.data])
            self.improved = self.Particle_performance < self.Particle_best_performance
            self.Particle_best_positions[self.improved] = self.Particle_Positions[self.improved]
            self.Particle_best_performance[self.improved] = self.Particle_performance[self.improved]
            self.best_index = np.argmin(self.Particle_performance)
            if self.Particle_performance[self.best_index] < self.Global_best_performance:
                self.Global_best_performance = self.Particle_performance[self.best_index]
                self.Global_best_position = self.Particle_Positions[self.best_index]
            self.r1, self.r2 = np.random.rand(), np.random.rand()
            self.Particle_Velocity = (
                self.w * self.Particle_Velocity +
                self.C1 * self.r1 * (self.Particle_best_positions - self.Particle_Positions) +
                self.C2 * self.r2 * (self.Global_best_position - self.Particle_Positions) +
                self.Particle_Velocity * 0.001 * (np.random.rand(*self.Particle_Velocity.shape) - 0.5) * 2
            )
            self.Last_global_best = self.Global_best_performance
            if abs(self.Last_global_best - self.Global_best_performance) < 0.001:
                self.no_change_count += 1
            else:
                self.no_change_count = 0
            if self.no_change_count > 50:
                self.no_change_count = 0
                self.Particle_Positions = np.array([self.Global_best_position for _ in range(self.Particles)])
                self.Particle_Velocity = np.array([(np.random.randn(*self.v.shape) - 0.5) * 0.1 for _ in range(self.Particles)])
        self.Objective_function([self.t, [self.Global_best_position[:, 0], self.Global_best_position[:, 1]], self.k], True)

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_ylabel("Dimensionless surface exposure")
        self.ax.set_xlabel("Foil position")
        self.ax.set_ylim([0, 50])
        self.ax.axhline(22, 0, 207)
        self.ax.plot(self.surface_exposure)
        plt.show()

        self.fig, self.ax = plt.subplots()
        plt.title("Optimum solution")
        self.ax.set_aspect('equal')
        self.ax.plot(self.Tip_points[:, 0], self.Tip_points[:, 1])
        self.nodes = interpolate.splev(np.linspace(0, 1, 200, endpoint=False), [self.t, [self.Global_best_position[:, 0], self.Global_best_position[:, 1]], self.k])
        self.ax.plot(self.nodes[0], self.nodes[1])
        self.ax.plot(self.offset_points[:, 0], self.offset_points[:, 1])
        plt.show()

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.loss_array, c="Black")
        plt.title("Optimisation loss")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        plt.show()

        self.trajectory = Trajectory(self.Tip_points, self.trajectory_nodes)
        self.surface_exposure = self.trajectory.predict_sde()
        self.fig, self.ax = plt.subplots(figsize=(8, 3), dpi=100)
        self.ax.plot(self.surface_exposure, c='black')
        self.ax.set_xlabel("Surface panel index")
        self.ax.set_ylabel("Dimensionless heating parameter")
        self.ax.axhline(22, -20, 220, color="Red")
        plt.tight_layout()
        self.ax.grid()
        plt.show()

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_title("Tip foil profile and inital wire trajectory")
        self.ax.set_ylabel("Y position [mm]")
        self.ax.set_xlabel("X position [mm]")
        self.ax.plot(self.Tip_points[:, 0], self.Tip_points[:, 1], color="black")
        self.nodes = interpolate.splev(np.linspace(0, 1, 200, endpoint=False), [self.t, [self.Global_best_position[:, 0], self.Global_best_position[:, 1]], self.k])
        self.line, = self.ax.plot(self.nodes[0], self.nodes[1], color="red")
        self.add_arrow_to_line2D(self.ax, self.line, arrow_locs=np.linspace(0., 1., 200), arrowstyle='->')
        self.line, = self.ax.plot(self.offset_points[:, 0], self.offset_points[:, 1], color="blue")
        self.add_arrow_to_line2D(self.ax, self.line, arrow_locs=np.linspace(0., 1., 200), arrowstyle='->')


        plt.show()


    def Export(self):
        self.nodes = interpolate.splev(np.linspace(0, 1, 200, endpoint=False), [self.t, [self.Global_best_position[:, 0], self.Global_best_position[:, 1]], self.k])
        self.nodes = np.dstack((self.nodes[0], self.nodes[1]))[0]
        print(np.max(self.nodes, axis=0))
        self.cutting_nodes = self.nodes - np.max(self.nodes, axis=0)
        self.output = open("g_code.txt", "w")
        self.coding = "G90\nM3\nG1 X0 Y0 A0 B0 F600\nG1 X0 Y-10 A0 B-10 F200\nG92 X0 Y0 A0 B0\n"
        for node in self.cutting_nodes:
            self.coding += f"G1 X{round(node[0], 4)} Y{round(node[1], 4)} A{round(node[0], 4)} B{round(node[1], 4)} F200\n"
        self.end_command = "M3\nG1 X0 A0 F600\n"
        print(self.coding)
        self.coding += self.end_command
        self.output.write(self.coding)
        self.output.close()

if __name__ == "__main__":
    main()
