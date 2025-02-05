import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

time = ctrl.Antecedent(np.arange(0, 24, 0.5), 'time')
hours_worked = ctrl.Antecedent(np.arange(0, 12, 0.5), 'hours_worked')
sleep = ctrl.Consequent(np.arange(5, 14, 0.25), 'sleep')

time['day'] = fuzz.trapmf(time.universe, [8,10,12,16])
time['evening'] = fuzz.trapmf(time.universe, [14, 18, 19, 20])
time['late'] = fuzz.smf(time.universe, 19, 24)

hours_worked['short'] = fuzz.zmf(hours_worked.universe, 0, 6)
hours_worked['normal'] = fuzz.trapmf(hours_worked.universe, [5, 6, 7, 8])
hours_worked['long'] = fuzz.smf(hours_worked.universe, 7, 12)

sleep['short'] = fuzz.zmf(sleep.universe, 5, 7)
sleep['normal'] = fuzz.trapmf(sleep.universe, [6, 7, 8, 8])
sleep['long'] = fuzz.smf(sleep.universe, 8, 12)

# time.view()
# hours_worked.view()
# sleep.view()

rule1 = ctrl.Rule(time['late'] | hours_worked['long'], sleep['long'])
rule2 = ctrl.Rule(hours_worked['normal'], sleep['normal'])
rule3 = ctrl.Rule(time['day'] & hours_worked['short'], sleep['short'])

sleep_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
sleep_sim = ctrl.ControlSystemSimulation(sleep_ctrl)

sleep_sim.input['time'] = 12
sleep_sim.input['hours_worked'] = 10

sleep_sim.compute()

print (sleep_sim.output['sleep'])