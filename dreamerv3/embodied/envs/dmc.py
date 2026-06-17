import functools
import os

import elements
import embodied
import numpy as np
from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

from . import from_dm


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), proprio=True, image=True, camera=-1,
      norender=False):
    # norender=True (with image=False) makes the env fully render-free: no
    # physics.render() call at all. Use for proprioceptive-only studies to
    # avoid the EGL/MuJoCo render path. Default keeps stock behavior.
    self._norender = norender
    if not (norender and not image) and 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        env = manipulation.load(task + '_vision')
      elif domain == 'rodent':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        env = getattr(basic_rodent_2020, task)()
      else:
        env = suite.load(domain, task)
    self._dmenv = env
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._proprio = proprio
    self._image = image
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = self._env.obs_space.copy()
    if not self._proprio:
      spaces = {k: spaces[k] for k in basic}
    # Render-free proprio mode: when image rendering is disabled entirely
    # (image=False AND no log image wanted), skip the image space so the env
    # never calls physics.render() — avoids the EGL/MuJoCo render path.
    if self._image:
      spaces['image'] = elements.Space(np.uint8, self._size + (3,))
    elif not self._norender:
      spaces['log/image'] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    if not self._proprio:
      obs = {k: obs[k] for k in basic}
    if self._image:
      obs['image'] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    elif not self._norender:
      obs['log/image'] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    # else: render-free proprio mode — no physics.render() call.
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
