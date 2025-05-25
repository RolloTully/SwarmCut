# SwarmCut - Trajectory optimiser for Hotwire cutting

## Abstract
The performance of Hotwire tool paths is difficult to calculate due to the nonlinear and multivariable interactions with the kerf width. Accurate cutting requires a balance of trajectory, velocity, and heating to achieve high geometric fidelity. To reduce the complexity we recontextualise the problem. Instead of attempting to predict wire kerf, a problem which has been explored thoroughly but has failed to see real world use, we instead consider it as an energy distribution problem. We introduce a new parameter, Critical Specific Energy (CSE), that quantifies the minimum thermal energy that must be delivered to the surface of a material to achieve ablation. Use of CSE simplifies the thermal-mechanical interaction, and allows for the construction of an indirect optimisation problem.  We formulate a bespoke Particle Swarm Optimisation (PSO) meta-heuristic using CSE to minimise surface deviation while considering real world manufacturing constraints. This method termed Swarm-Cut, is applied to a composite sandwich panel wing core, using Swarm-Cut reduced the mean surface deviation from 1.67 mm to 0.53 mm (68.3% reduction) and reduced the standard deviation of surface deviation(σΔ) from 1.59 mm to 0.352 mm (77.9% reduction) when compared to baseline trajectories demonstrating  gains in both manufacturing consistency, accuracy and simplifying process tuning.

## Results:
![Alt text](/Figures/Path_opt.png)
SwarmCut has been used to manufacture aircraft at UOM UAV and the results produced show a consistent improvement in manufacturing accuracy.
In the plot above you can see the Original offset path in blue, and the optimised path in red. If you follow the path you can see the separation between the paths changes through the cut.
## Installation:

### Windows + Python 3.10

In windows Installation can be done via conda
```
git clone https://github.com/RolloTully/SwarmCut.git
cd SwarmCut/SwarmCut
conda env create --name SwarmCut --file=environments.yml
```

## Usage:
### Step 1: Define Geometry
Foil Geometry is defined by a standard X,Y foil definition.

You can define your geometry by altering the foil variable
```
self.foil_addr = "\\Path to your\\airfoil.dat"
self.Chord = your_chord_length
```

### Step 2: Material definition
If you know your system CSE value
```
self.critical_surface_energy = your_material_cse
```

### Step 3: Optimisation
Once the your parameters have been set run:
```
python main.py
```
The program will start by initialising the optimiser.
Close all the plots(will remove these later)
The optimiser will take a while to run but once its done you don't need to do anything further

### Step 4: Export
The program will output "g_code.txt", this contains the 4 axis tool path for manufacturing

## CSE Calibration
If you don't know you system CSE value you will need to calibrate the manufacturing process.

This system is relatively simple. For EPP foams start with CSE = 22, run the optimisation once manufacturing is complete measure a reference dimension and compare it to the design dimension.
If you part is over sized gradually increase your CSE value until the part reached the desired size, this is your system CSE value.

## How does it work?
