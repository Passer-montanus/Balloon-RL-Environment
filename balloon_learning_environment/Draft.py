import dataclasses
import datetime as dt
import math
import jax.random
import numpy as np
from balloon_learning_environment.utils import units
import time

def _rolling( key, sizes: int) -> int:
    res = jax.random.uniform(key, (), minval=0, maxval=sizes)
    return math.floor(res)


print(_rolling(key=jax.random.PRNGKey(int(time.time() * 1e6)), sizes=2))