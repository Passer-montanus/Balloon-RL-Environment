# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A renderer using matplotlib.

This is not the most efficient renderer, but it is convenient.
"""

from typing import Iterable, Optional, Text, Union
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env import balloon_arena
from balloon_learning_environment.env import forbidden_area
from balloon_learning_environment.env.rendering import renderer
from flax.metrics import tensorboard
from matplotlib import dates as mdates
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import art3d
import numpy as np
from balloon_learning_environment.utils import units
import matplotlib
matplotlib.use('Qt5Agg')

class MatplotlibRenderer(renderer.Renderer):
  """Contains functions for rendering the simulator state with matplotlib."""

  def __init__(self):
    self.reset()

    self._x_lims = (-150.0, 150.0)  # km
    self._y_lims = (-150.0, 150.0)  # km
    self._altitude_lims = (15, 20)  # km
    self._target_x = 0.0
    self._target_y = 0.0
    self._target_radius = 50.0  # TODO(joshgreaves): Get this from the env.

    self._fig = plt.figure(figsize=(15, 10))
    spec = gridspec.GridSpec(ncols=1,
                             nrows=1,
                             height_ratios=[1])
    self._axes = [self._fig.add_subplot(spec[0], projection='3d')]
    self._axes.append(
        inset_axes(self._axes[0],
                   width='50.0%',
                   height='10.0%',
                   loc='upper right'))
    self._axes.append(
        inset_axes(self._axes[0],
                   width='37.5%',
                   height='37.5%',
                   loc='upper left'))
    self._axes.append(
        inset_axes(self._axes[0],
                   width='15%',
                   height='25%',
                   loc='lower right'))


  def reset(self) -> None:
    self._trajectory = list()
    self._charge = list()
    self._datetime = list()
    self.Farea_vector1 = [75, 75, 60]
    self.Farea_vector1 = [75, 75, 60]
    self.Farea_vector1 = [75, 75, 60]
    self.obs = np.zeros((1105,),dtype=np.float32)

  def step(self, state: simulator_data.SimulatorState,obs:np.ndarray) -> None:
    balloon_state = state.balloon_state
    altitude = state.atmosphere.at_pressure(balloon_state.pressure).height
    self._charge.append(balloon_state.battery_soc * 100.0)
    self._datetime.append(balloon_state.date_time)
    self.Farea_vector1=[balloon_state.Farea1_x, balloon_state.Farea1_y, balloon_state.Farea1_r]
    self.Farea_vector2=[balloon_state.Farea2_x, balloon_state.Farea2_y, balloon_state.Farea2_r]
    self.Farea_vector3=[balloon_state.Farea3_x, balloon_state.Farea3_y, balloon_state.Farea3_r]
    self.obs=obs
    self._trajectory.append(
        np.asarray([balloon_state.x.kilometers,
                    balloon_state.y.kilometers,
                    altitude.kilometers]))



  def render(self,
             mode: Text,
             summary_writer: Optional[tensorboard.SummaryWriter] = None,
             iteration: Optional[int] = None) -> Union[None, np.ndarray, Text]:
    """Renders a frame.

    Args:
      mode: One of `human`, or `rgb_array`. `human` corresponds to rendering
        directly to the screen. `rgb_array` renders to a numpy array and returns
        it. This renderer doesn't support the `ansi` mode.
      summary_writer: If not None and mode == 'tensorboard', will also render
        the image to the tensorboard summary.
      iteration: Iteration number used for writing to tensorboard.

    Returns:
      None or a numpy array of rgb data depending on the mode.
    """
    if mode not in self.render_modes:
      raise ValueError('Unsupported render mode {}. Use one of {}.'.format(
          mode, self.render_modes))

    for ax in self._axes:
      ax.clear()
    flight_path = np.vstack(self._trajectory)
    self._plot_3d_flight_path(flight_path,Farea_matrix=[self.Farea_vector1, self.Farea_vector2, self.Farea_vector3])
    self._plot_inset(flight_path,Farea_matrix=[self.Farea_vector1, self.Farea_vector2, self.Farea_vector3])
    self._plot_power()
    self._plot_Obs_Farea(obs=self.obs,Farea_matrix=[self.Farea_vector1, self.Farea_vector2, self.Farea_vector3])

    if mode == 'human':
      plt.pause(0.001)  # Renders the image and runs the GUI loop.
    elif mode == 'rgb_array' or mode == 'tensorboard':
      self._fig.canvas.draw()
      rgb_string = self._fig.canvas.tostring_rgb()
      width, height = self._fig.canvas.get_width_height()
      frame = np.frombuffer(
          rgb_string, dtype=np.uint8).reshape(height, width, -1)
      if mode == 'rgb_array':
        return frame

      if summary_writer is not None and iteration is not None:
        summary_writer.image('Balloon/Path', frame, iteration)
        summary_writer.flush()
        plt.cla()

  @property
  def render_modes(self) -> Iterable[Text]:
    return ['human', 'rgb_array', 'tensorboard']

  def _plot_3d_flight_path(self, flight_path: np.ndarray,Farea_matrix:np.ndarray):
    ax = self._axes[0]

    # TODO(joshgreaves): Longitude/Latitude isn't quite right
    ax.set_xlabel('Longitude displacement (km)')
    ax.set_ylabel('Latitude displacement (km)')
    ax.set_zlabel('Altitude (km)')

    ax.set_xlim(self._x_lims)
    ax.set_ylim(self._y_lims)
    ax.set_zlim(self._altitude_lims)
    ax.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.grid(True)

    # Draw the target radius.
    ax.scatter3D(
        self._target_x,
        self._target_y,
        self._altitude_lims[0],
        c='k',
        lw=1.0,
        marker='x',
        s=100)
    circle = plt.Circle((self._target_x, self._target_y),
                        self._target_radius,
                        edgecolor='k',
                        ls='--',
                        fill=False)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=self._altitude_lims[0], zdir='z')

    #Plot the Forbiddent Area1
    Farea_state1 = Farea_matrix[0]

    ax.scatter3D(
        Farea_state1[0],
        Farea_state1[1],
        self._altitude_lims[0],
        c='r',
        lw=1.0,
        marker='x',
        s=25)

    Farea_circle = plt.Circle((Farea_state1[0], Farea_state1[1]),
                        Farea_state1[2],
                        edgecolor='g',
                        ls='--',
                        fill=False)
    ax.add_patch(Farea_circle)
    art3d.pathpatch_2d_to_3d(Farea_circle, z=self._altitude_lims[0], zdir='z')
    Farea_circle_w = plt.Circle((Farea_state1[0], Farea_state1[1]),
                              Farea_state1[2]*14/30,
                              edgecolor='b',
                              ls='-.',
                              fill=False)
    ax.add_patch(Farea_circle_w)
    art3d.pathpatch_2d_to_3d(Farea_circle_w, z=self._altitude_lims[0], zdir='z')
    Farea_circle_s = plt.Circle((Farea_state1[0], Farea_state1[1]),
                                Farea_state1[2]*4/30,
                                edgecolor='r',
                                ls='-',
                                fill=False)
    ax.add_patch(Farea_circle_s)
    art3d.pathpatch_2d_to_3d(Farea_circle_s, z=self._altitude_lims[0], zdir='z')


    # Plot the Forbiddent Area2
    Farea_state2 = Farea_matrix[1]

    ax.scatter3D(
        Farea_state2[0],
        Farea_state2[1],
        self._altitude_lims[0],
        c='r',
        lw=1.0,
        marker='x',
        s=25)

    Farea_circle1 = plt.Circle((Farea_state2[0], Farea_state2[1]),
                              Farea_state2[2],
                              edgecolor='g',
                              ls='--',
                              fill=False)
    ax.add_patch(Farea_circle1)
    art3d.pathpatch_2d_to_3d(Farea_circle1, z=self._altitude_lims[0], zdir='z')
    Farea_circle1_w = plt.Circle((Farea_state2[0], Farea_state2[1]),
                                Farea_state2[2]*14/30,
                                edgecolor='b',
                                ls='-.',
                                fill=False)
    ax.add_patch(Farea_circle1_w)
    art3d.pathpatch_2d_to_3d(Farea_circle1_w, z=self._altitude_lims[0], zdir='z')
    Farea_circle1_s = plt.Circle((Farea_state2[0], Farea_state2[1]),
                                Farea_state2[2]*4/30,
                                edgecolor='r',
                                ls='-',
                                fill=False)
    ax.add_patch(Farea_circle1_s)
    art3d.pathpatch_2d_to_3d(Farea_circle1_s, z=self._altitude_lims[0], zdir='z')

    # Plot the Forbiddent Area2
    Farea_state3 = Farea_matrix[2]

    ax.scatter3D(
        Farea_state3[0],
        Farea_state3[1],
        self._altitude_lims[0],
        c='r',
        lw=1.0,
        marker='x',
        s=25)

    Farea_circle2 = plt.Circle((Farea_state3[0], Farea_state3[1]),
                               Farea_state3[2],
                               edgecolor='g',
                               ls='--',
                               fill=False)
    ax.add_patch(Farea_circle2)
    art3d.pathpatch_2d_to_3d(Farea_circle2, z=self._altitude_lims[0], zdir='z')
    Farea_circle2_w = plt.Circle((Farea_state3[0], Farea_state3[1]),
                                 Farea_state3[2]*14/30,
                                 edgecolor='b',
                                 ls='-.',
                                 fill=False)
    ax.add_patch(Farea_circle2_w)
    art3d.pathpatch_2d_to_3d(Farea_circle2_w, z=self._altitude_lims[0], zdir='z')
    Farea_circle2_s = plt.Circle((Farea_state3[0], Farea_state3[1]),
                                 Farea_state3[2]*4/30,
                                 edgecolor='r',
                                 ls='-',
                                 fill=False)
    ax.add_patch(Farea_circle2_s)
    art3d.pathpatch_2d_to_3d(Farea_circle2_s, z=self._altitude_lims[0], zdir='z')


    # Draw the trajectory
    ax.plot3D(flight_path[:, 0],
              flight_path[:, 1],
              flight_path[:, 2], c='C0')
    ax.scatter3D(
        flight_path[-1, 0],
        flight_path[-1, 1],
        flight_path[-1, 2],
        color='C0',
        lw=0.1,
        marker='o',
        s=100)

    # Add a stem to the balloon
    last_x = flight_path[-1, 0]
    last_y = flight_path[-1, 1]
    ax.plot3D([last_x, last_x],
              [last_y, last_y],
              [self._altitude_lims[0], flight_path[-1, 2]],
              color='C0')

  def _plot_power(self):
    ax = self._axes[1]

    ax.set_ylim([0.0, 110])
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Power (%)')
    ax.set_xticks([self._datetime[0], self._datetime[-1]])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.plot(self._datetime, self._charge, color='C0')

  def _plot_inset(self, flight_path: np.ndarray,Farea_matrix:np.ndarray):
    ax = self._axes[2]

    ax.set_xlim(self._x_lims)
    ax.set_ylim(self._y_lims)
    ax.set_xticks([self._x_lims[0], 0.0, self._x_lims[1]])
    ax.set_yticks([self._y_lims[0], 0.0, self._y_lims[1]])
    circle = plt.Circle([self._target_x, self._target_y],
                        self._target_radius,
                        edgecolor='k',
                        ls='--',
                        fill=False)
    ax.add_patch(circle)
        #Plot forbiddent area1
    Farea_state1 = Farea_matrix[0]
    Farea_circle = plt.Circle([Farea_state1[0], Farea_state1[1]],
                        Farea_state1[2],
                        edgecolor='g',
                        ls='--',
                        fill=False)
    ax.add_patch(Farea_circle)
    Farea_circle_w = plt.Circle([Farea_state1[0], Farea_state1[1]],
                              Farea_state1[2] * 14/30,
                              edgecolor='b',
                              ls='-.',
                              fill=False)
    ax.add_patch(Farea_circle_w)
    Farea_circle_s = plt.Circle([Farea_state1[0], Farea_state1[1]],
                                Farea_state1[2] * 4 / 30,
                                edgecolor='r',
                                ls='-',
                                fill=False)
    ax.add_patch(Farea_circle_s)
    ax.plot(flight_path[:, 0], flight_path[:, 1], color='C0')
    ax.scatter(flight_path[-1, 0], flight_path[-1, 1], color='C0')
    # Plot forbiddent area2
    Farea_state2 = Farea_matrix[1]
    Farea_circle1 = plt.Circle([Farea_state2[0], Farea_state2[1]],
                              Farea_state2[2],
                              edgecolor='g',
                              ls='--',
                              fill=False)
    ax.add_patch(Farea_circle1)
    Farea_circle1_w = plt.Circle([Farea_state2[0], Farea_state2[1]],
                                Farea_state2[2] * 14 / 30,
                                edgecolor='b',
                                ls='-.',
                                fill=False)
    ax.add_patch(Farea_circle1_w)
    Farea_circle1_s = plt.Circle([Farea_state2[0], Farea_state2[1]],
                                Farea_state2[2] * 4 / 30,
                                edgecolor='r',
                                ls='-',
                                fill=False)
    ax.add_patch(Farea_circle1_s)
    ax.plot(flight_path[:, 0], flight_path[:, 1], color='C0')
    ax.scatter(flight_path[-1, 0], flight_path[-1, 1], color='C0')
    # Plot forbiddent area3
    Farea_state3 = Farea_matrix[2]
    Farea_circle2 = plt.Circle([Farea_state3[0], Farea_state3[1]],
                              Farea_state3[2],
                              edgecolor='g',
                              ls='--',
                              fill=False)
    ax.add_patch(Farea_circle2)
    Farea_circle2_w = plt.Circle([Farea_state3[0], Farea_state3[1]],
                                 Farea_state3[2] * 14 / 30,
                                 edgecolor='b',
                                 ls='-.',
                                 fill=False)
    ax.add_patch(Farea_circle2_w)
    Farea_circle2_s = plt.Circle([Farea_state3[0], Farea_state3[1]],
                                 Farea_state3[2] * 4 / 30,
                                 edgecolor='r',
                                 ls='-',
                                 fill=False)
    ax.add_patch(Farea_circle2_s)
    ax.plot(flight_path[:, 0], flight_path[:, 1], color='C0')
    ax.scatter(flight_path[-1, 0], flight_path[-1, 1], color='C0')
  #TODO Plot the observation of forbiddent area
  def _plot_Obs_Farea(self,obs:np.ndarray, Farea_matrix:np.ndarray):
    ax = self._axes[3]
    ax.set_xlim([0.0, 110])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(2, 0.05, 'OBS')
    ax.text(2,0.04,'Farea_dist1:')
    ax.text(2, 0.03, obs[16])
    ax.text(2, 0.02, 'Farea_dist1_r')
    ax.text(2, 0.01, obs[17])
    ax.text(2, 0, 'Farea_dist2')
    ax.text(2, -0.01, obs[18])
    ax.text(2, -0.02, 'Farea_dist2_r')
    ax.text(2, -0.03, obs[19])
    ax.text(2, -0.04, 'Farea_dist3')
    ax.text(2, -0.05, obs[20])
    ax.text(2, -0.06, 'Farea_dist3_r')
    ax.text(2, -0.07, obs[21])
    ax.plot()