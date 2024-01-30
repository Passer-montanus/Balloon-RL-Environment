from balloon_learning_environment.env import forbidden_area
from balloon_learning_environment.env import simulator_data

from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon import balloon
import math


# If the balloon stay in station keeping area
if dist_T <= dist_R:
    reward = 1
elif Farea_R < Farea_dist <= Buffer_R:
    reward = reward_dropoff * math.exp(-0.69314718056 / reward_halflife * (dist_T - dist_R).kilometers)
    reward -= SumOfFarea(punish_dropoff * math.exp(-0.69314718056 / punish_halflife * (Farea_dist - Farea_R).kilometers))
elif Farea_dist <= Farea_R:
    reward = -2
else:
    reward = reward_dropoff * math.exp(-0.69314718056 / reward_halflife * (dist_T - dist_R).kilometers)

