import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

conditions = ctrl.Antecedent(np.arange(0, 5, 0.25), 'conditions')
visibility = ctrl.Antecedent(np.arange(0, 101, 1), 'visibility')
speed = ctrl.Consequent(np.arange(0, 60, 1), 'speed')

# Custom membership functions
conditions['slight'] = fuzz.zmf(conditions.universe, 1, 1.25)
conditions['intermediate'] = fuzz.trapmf(conditions.universe, [1, 1.25, 2.5, 3])
conditions['rough'] = fuzz.smf(conditions.universe, 2.5, 3)

visibility['bad'] = fuzz.zmf(visibility.universe, 25, 60)
visibility['ok'] = fuzz.trapmf(visibility.universe, [40, 60, 70, 80])
visibility['good'] = fuzz.smf(visibility.universe, 70, 80)

speed['low'] = fuzz.zmf(speed.universe, 15, 20)
speed['medium'] = fuzz.trapmf(speed.universe, [15, 30, 40, 50])
speed['high'] = fuzz.smf(speed.universe, 45, 50)

# Should you wish to view any of the membership functions
conditions.view()

rule1 = ctrl.Rule(visibility['bad'] | conditions['rough'], speed['low'])
rule2 = ctrl.Rule(visibility['ok'], speed['medium'])
rule3 = ctrl.Rule(visibility['good'] & conditions['slight'], speed['high'])

speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
speed_sim = ctrl.ControlSystemSimulation(speed_ctrl)

speed_sim.input['conditions'] = 1.1
speed_sim.input['visibility'] = 55

speed_sim.compute()

print (speed_sim.output['speed'])

# If you want to view the centroid
speed.view(sim=speed_sim)