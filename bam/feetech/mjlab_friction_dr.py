# Copyright 2026 ftservo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-episode friction domain randomization and encoder-through-backlash feedback.

Companion module to :mod:`bam.mjlab` — same optional dependency group
(``pip install better-actuator-models[mjlab]``).


Where this lives
----------------
``bam/feetech/mjlab_friction_dr.py`` — next to the Feetech drivers it is most
often paired with, but deliberately **not** re-exported from
``bam/feetech/__init__.py``, which stays empty on purpose. The module is
self-contained (absolute ``bam.*`` imports only), and
``scripts/verify_mjlab_friction_dr.py`` looks for it **only in the driver
directory of the servo under test** — the directory holding the ``actuator.py``
that defines that servo, read out of ``bam/actuators.py`` without importing it.
So the location is load-bearing for that harness: an ``xl330`` run aborts here
instead of silently verifying this file, and covering another family means
putting a copy beside that family's driver. This module pulls in ``torch``,
``mujoco``, ``mujoco_warp`` and ``mjlab`` (through
:class:`bam.mjlab.BamActuator`), so a plain ``import bam.feetech`` on a machine
without the ``[mjlab]`` extra must keep working; only an explicit full-path
import loads the GPU simulation stack::

    from bam.feetech.mjlab_friction_dr import FrictionDRBamActuatorCfg

The classes are **not** Feetech-specific: they work with any
``DCMotorActuator`` (STS3215, HD1910, XL330, …). Nothing here inspects the servo
model — the friction randomization reads the BAM ``Model`` friction budget and
the firmware-state workaround only touches ``q_target_smooth`` if it exists.

The plain-MuJoCo counterpart of this module is its sibling
:mod:`bam.feetech.mujoco_friction_dr`, which does the same two jobs for
:class:`bam.mujoco.MujocoController` and needs neither ``torch`` nor the
``[mjlab]`` extra.


What upstream 1.0.2 already provides — do NOT re-implement
----------------------------------------------------------
1. ``BamActuatorCfg.friction_scale_range`` — a per-env friction multiplier,
   ``frictionloss *= friction_scale`` right before it is written into MuJoCo
   (``bam/mjlab.py`` ``_compute_friction_budget``). Sampled in ``initialize()``
   and explicitly **held constant across resets** (see ``BamActuator.reset``).
2. Lazy ``q_target_smooth`` init in ``STS3215Actuator.__init__`` /
   ``compute_control`` — so the old "seed the buffer or it raises
   ``AttributeError``" workaround is **no longer needed** on this version.
3. ``bam_init`` startup event — expands ``dof_frictionloss`` / ``dof_damping``
   per world. Required for any BamActuator; ``BamActuator._write_frictions``
   raises a clear ``RuntimeError`` if you forget it.


What this module adds
---------------------
1. **Per-episode friction re-sampling.** Upstream's ``friction_scale_range``
   draws one value per env for the whole training run (like mass DR). That is a
   fixed population of N virtual robots, not a changing one. Standard domain
   randomization re-draws per episode, which this module adds via
   ``per_episode_friction_scale_range`` — either sampled automatically in
   ``reset()`` or driven from a task-config reset event through
   :meth:`FrictionDRBamActuator.set_friction_scale`.

2. **Encoder reads through the gear backlash.** ``BacklashEncoderBamActuator``
   feeds the firmware position loop ``qpos[servo] + qpos[backlash]`` instead of
   ``qpos[servo]``, reproducing a servo whose magnetic encoder sits on the
   output side of the play: while the servo winds through the dead zone the
   measured position — and hence the PD error — does not change.

3. **Per-env firmware-state reset** (works around an upstream defect, see
   :meth:`FrictionDRBamActuator.reset`). ``BamActuator.initialize`` calls
   ``bam.reset()`` with the comment "Drop any internal firmware state left over
   from a previous sim", but ``bam.actuator.Actuator.reset`` is a bare
   ``pass`` and no actuator overrides it — ``stateful`` is ``False`` everywhere
   in the package and ``get_state``/``set_state`` are unimplemented stubs. So
   ``STS3215Actuator.q_target_smooth`` — which *is* control-law state — survives
   an episode reset. Since the joint teleports but the rate-limited internal
   goal does not, every episode starts with the goal slewing in from the
   previous episode's value. With the 1.0.2 ``max_velocity`` default of
   7.0 rad/s and ``dt = 5 ms`` that is 0.035 rad per step, i.e. ~29 steps
   (~145 ms) to recover a 1 rad teleport.

.. note::
   The recommended *upstream* fix is to declare ``stateful = True`` on
   ``STS3215Actuator`` and implement ``get_state``/``set_state`` (returning and
   restoring ``q_target_smooth``), which is exactly the contract
   ``bam.mujoco.MujocoController`` already honours. Until then, pass
   ``reset_firmware_state_on_reset=True`` (the default) here.

   This workaround is a no-op for actuators without ``q_target_smooth``
   (HD1910, XL330), so it is safe to enable unconditionally.

.. note::
   Verified end-to-end against a real mjlab / MuJoCo-Warp simulation by
   ``scripts/verify_mjlab_friction_dr.py`` (needs the ``[mjlab]`` extra). The
   servo under test is a required command-line argument there: ``--json-path``
   names the params JSON and the class is read from its ``"actuator"`` key, so
   the same harness covers STS3215, HD1910 and any other registered motor
   (``--list-params`` prints what is bundled). It looks for this file beside the
   driver of the servo under test, so moving the module to another family
   directory changes which servos it can run — and a servo whose family has no
   copy aborts rather than silently importing this one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
    "FrictionDRBamActuator",
    "FrictionDRBamActuatorCfg",
    "BacklashEncoderBamActuator",
    "BacklashEncoderBamActuatorCfg",
    "randomize_bam_friction",
]


class FrictionDRBamActuator(BamActuator):
    """``BamActuator`` with a per-env, re-samplable friction budget scale.

    The scale multiplies BAM's **entire** velocity-independent friction budget
    (Coulomb + Stribeck + load-dependent), which is the term that carries the
    dominant sim2real friction uncertainty (stiction / gearbox). The viscous
    (velocity-proportional) term goes to ``dof_damping`` untouched — scale it
    too by overriding :meth:`bam.mjlab.BamActuator.compute` if ever needed.

    The multiplication happens *before* the budget is written into
    ``dof_frictionloss``, so MuJoCo's constraint solver still performs the
    static-friction clipping (BAM Algorithm 1). Adding a passive friction
    torque to the returned motor torque instead would bypass that clipping.
    """

    cfg: "FrictionDRBamActuatorCfg"

    def initialize(self, mj_model, model, data, device) -> None:  # noqa: ANN001
        super().initialize(mj_model, model, data, device)

        # Upstream leaves friction_scale as None when friction_scale_range is
        # unset. This subclass always needs a live (N, 1) tensor so that the
        # per-episode sampler (and the env-side reset event) has somewhere to
        # write. An all-ones tensor is a numerical no-op.
        if self.friction_scale is None:
            self.friction_scale = torch.ones(
                self._num_envs, 1, dtype=torch.float32, device=device
            )
        # Baseline restored by reset_friction_scale, so per-episode sampling can
        # never accumulate across resets.
        self.default_friction_scale = self.friction_scale.clone()

        self._pending_firmware_reset: torch.Tensor | None = None

        per_ep = self.cfg.per_episode_friction_scale_range
        print(
            f"[FrictionDRBamActuator] joints={len(self._target_ids_list)} "
            f"envs={self._num_envs} "
            f"startup_scale_range={self.cfg.friction_scale_range} "
            f"per_episode_scale_range={per_ep} "
            f"reset_firmware_state={self.cfg.reset_firmware_state_on_reset}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-env friction scale API
    # ─────────────────────────────────────────────────────────────────────────

    def _env_ids_tensor(self, env_ids) -> torch.Tensor:  # noqa: ANN001
        """Normalize the several accepted ``env_ids`` forms to a long tensor."""
        if env_ids is None or env_ids is Ellipsis:
            return torch.arange(
                self._num_envs, dtype=torch.long, device=self._device
            )
        if isinstance(env_ids, slice):
            return torch.arange(
                self._num_envs, dtype=torch.long, device=self._device
            )[env_ids]
        ids = torch.as_tensor(env_ids, device=self._device)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).squeeze(-1)
        return ids.long().flatten()

    def set_friction_scale(self, env_ids, friction_scale) -> None:  # noqa: ANN001
        """Write an explicit friction scale for the given environments.

        ``friction_scale`` is broadcast against the selected envs, so a
        ``(len(env_ids), 1)`` tensor or a scalar both work.

        .. important::
            Call :meth:`reset_friction_scale` first (or let
            ``per_episode_friction_scale_range`` do it) — BAM's budget is
            multiplied, not assigned, so a fresh sample must start from the
            1.0 baseline rather than from the previous episode's value.
        """
        assert self.friction_scale is not None
        ids = self._env_ids_tensor(env_ids)
        scale = torch.as_tensor(
            friction_scale, dtype=torch.float32, device=self._device
        )
        self.friction_scale[ids] = scale

    def reset_friction_scale(self, env_ids=None) -> None:  # noqa: ANN001
        """Restore the startup baseline scale for the given environments.

        The baseline is ``1.0`` when ``friction_scale_range`` is unset, and the
        value drawn at startup when it is set (see ``initialize``).
        """
        assert self.friction_scale is not None
        ids = self._env_ids_tensor(env_ids)
        self.friction_scale[ids] = self.default_friction_scale[ids]

    def sample_friction_scale(self, env_ids=None) -> torch.Tensor:  # noqa: ANN001
        """Re-draw the per-episode scale from ``per_episode_friction_scale_range``.

        Returns the sampled ``(len(env_ids), 1)`` tensor. Non-accumulating:
        restores the baseline first.
        """
        assert self.friction_scale is not None
        lo, hi = self.cfg.per_episode_friction_scale_range  # type: ignore[misc]
        ids = self._env_ids_tensor(env_ids)
        self.reset_friction_scale(ids)
        samples = torch.rand(
            len(ids), 1, dtype=torch.float32, device=self._device
        ) * (hi - lo) + lo
        self.friction_scale[ids] = samples
        return samples

    # ─────────────────────────────────────────────────────────────────────────
    # Per-env firmware-state reset (upstream defect workaround)
    # ─────────────────────────────────────────────────────────────────────────

    def _mark_firmware_reset_pending(self, ids: torch.Tensor) -> None:
        pending = self._pending_firmware_reset
        if pending is None:
            self._pending_firmware_reset = ids
        else:
            self._pending_firmware_reset = torch.cat([pending, ids]).unique()

    def _apply_pending_firmware_reset(self, cmd: ActuatorCmd) -> None:
        """Re-seed the rate limiter from the current joint position.

        Mirrors what ``STS3215Actuator.compute_control`` does on its first call
        (``if self.q_target_smooth is None: self.q_target_smooth = q``), but for
        a subset of environments and without losing the rest. Deferred to
        ``compute()`` because the joint state is not available in ``reset()``.
        """
        if self._pending_firmware_reset is None:
            return
        act = self._bam_model.actuator
        smooth = getattr(act, "q_target_smooth", None)
        ids = self._pending_firmware_reset
        self._pending_firmware_reset = None

        if smooth is None:
            # No state yet: compute_control will seed it from q for every env.
            return
        if isinstance(smooth, torch.Tensor) and smooth.shape == cmd.pos.shape:
            with torch.no_grad():
                smooth[ids] = cmd.pos[ids]
            return
        # Stale state from a previous sim (shaped after the old (N, J)) — drop it.
        act.q_target_smooth = None

    # ─────────────────────────────────────────────────────────────────────────
    # mjlab lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, env_ids=None) -> None:  # noqa: ANN001
        """Episode reset: re-sample friction and re-seed the firmware state.

        Called by ``Entity.reset`` at episode boundaries and on mid-episode
        teleport commands (see ``mjlab.entity.Entity.reset``), *before* the
        ``mode="reset"`` event terms run. So if a task also registers an env-side
        friction event (:func:`randomize_bam_friction`), that event is applied
        last and wins — configure exactly one of the two.
        """
        super().reset(env_ids)
        ids = self._env_ids_tensor(env_ids)

        if self.cfg.reset_firmware_state_on_reset:
            self._mark_firmware_reset_pending(ids)

        if self.cfg.per_episode_friction_scale_range is not None:
            self.sample_friction_scale(ids)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        self._apply_pending_firmware_reset(cmd)
        return super().compute(cmd)


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    """``BamActuatorCfg`` plus per-episode friction randomization.

    :param per_episode_friction_scale_range: If set, a fresh per-env friction
        scale is sampled from this range on **every episode reset** (e.g.
        ``(0.9, 1.1)``). Mutually exclusive in spirit with the inherited
        ``friction_scale_range`` (startup-only): configure one, not both. If both
        are set the startup draw defines the baseline and each episode
        overwrites it.
    :param reset_firmware_state_on_reset: Re-seed stateful control-law buffers
        (``q_target_smooth``) for the resetting environments. Works around the
        unimplemented ``Actuator.reset`` in 1.0.2 — see the module docstring.
        Harmless for actuators that hold no such state.
    """

    per_episode_friction_scale_range: tuple[float, float] | None = None
    reset_firmware_state_on_reset: bool = True

    def build(
        self,
        entity: "Entity",
        target_ids: list[int],
        target_names: list[str],
    ) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    """``FrictionDRBamActuator`` whose firmware PD reads the encoder THROUGH backlash.

    Backlash models put an unactuated ``passive_<joint>_backlash`` hinge in
    series with each servo joint: the servo joint is the motor output, the
    backlash joint is the play between it and the link, and the link angle is
    their sum.

    On the real servo the magnetic encoder sits on the OUTPUT side of that play,
    so the firmware position loop closes on ``main + backlash`` — while the
    servo winds through the dead zone the measured position (and hence the PD
    error) does not change. This subclass reproduces that: the ``pos`` fed to
    BAM's voltage control law becomes ``qpos[main] + qpos[backlash]``.

    ``vel`` is left motor-side on purpose: in BAM it drives back-EMF and
    friction, which are rotor physics, not an encoder-derived firmware signal.

    Degrades to a plain friction-DR actuator on models without backlash joints
    (per-joint mask), so it is safe to use on any robot model.

    .. note::
        The ``+`` above assumes the play hinge is on the **same body and the
        same axis direction** as the motor joint it follows (the contract
        ``add_backlash.py`` implements). A flipped or rotated axis would be
        silently wrong here, so pass the robot's MJCF to
        ``scripts/verify_mjlab_friction_dr.py --model-xml`` to have that checked
        per joint, along with full coverage (no hinge left without play).

    .. warning::
        Pair this with matching **observations**. If the policy observes the
        motor-side angle while the firmware closes on the encoder view, the
        policy is trained against a channel the real robot does not provide.
        Both must read ``qpos[servo] + qpos[backlash]``. Also exclude the
        ``passive_*_backlash`` joints from any soft-limit penalty reward (they
        legitimately ride their hard limits) and from regex-based joint
        selections that would otherwise match them ambiguously.
    """

    def initialize(self, mj_model, model, data, device) -> None:  # noqa: ANN001
        super().initialize(mj_model, model, data, device)
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids, mask = [], []
        for name in self._target_names:
            bl_id = name_to_local.get(f"passive_{name}_backlash")
            ids.append(0 if bl_id is None else bl_id)
            mask.append(0.0 if bl_id is None else 1.0)
        self._backlash_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._backlash_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        n_backlash = int(self._backlash_mask.sum().item())
        print(
            f"[BacklashEncoderBamActuator] encoder-through-backlash feedback on "
            f"{n_backlash}/{len(mask)} joints"
        )
        if n_backlash == 0:
            print(
                "[BacklashEncoderBamActuator] no passive_<joint>_backlash joints "
                "found — behaving as a plain FrictionDRBamActuator."
            )

    def get_command(self, data) -> ActuatorCmd:
        cmd = super().get_command(data)
        pos = cmd.pos + data.joint_pos[:, self._backlash_joint_ids] * self._backlash_mask
        return dataclasses.replace(cmd, pos=pos)


@dataclass(kw_only=True)
class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
    """``FrictionDRBamActuatorCfg`` whose PD feedback reads through backlash joints."""

    def build(
        self,
        entity: "Entity",
        target_ids: list[int],
        target_names: list[str],
    ) -> BacklashEncoderBamActuator:
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)


# ─────────────────────────────────────────────────────────────────────────────
# Optional env-side driver
# ─────────────────────────────────────────────────────────────────────────────


def randomize_bam_friction(
    env,
    env_ids,
    scale_range: tuple[float, float],
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Reset event: re-sample the friction scale of every friction-DR actuator.

    Use this when the range must live in the *task* config rather than in the
    actuator config — e.g. a curriculum that widens ``scale_range`` over
    training. It is applied after ``Entity.reset`` (see ``_reset_idx``), so it
    takes precedence over ``per_episode_friction_scale_range``; configure one or
    the other.

    Silently does nothing when the actuator is not friction-DR capable — which
    is the same failure mode as forgetting the event entirely, so prefer
    ``per_episode_friction_scale_range`` unless you actually need the task-side
    range: the actuator-level path raises at config time instead of going quiet.

    Typical registration::

        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=randomize_bam_friction,
            mode="reset",
            params={"scale_range": (0.9, 1.1)},
        )
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")

    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.as_tensor(env_ids, device=env.device).long().flatten()

    asset = env.scene[asset_cfg.name]
    matched = False
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            # Restore first, then sample: the scale is multiplicative, so
            # skipping this would accumulate across episodes.
            actuator.reset_friction_scale(ids)
            lo, hi = scale_range
            actuator.set_friction_scale(
                ids,
                torch.rand(len(ids), 1, device=env.device) * (hi - lo) + lo,
            )
            matched = True

    if not matched:
        print(
            f"[randomize_bam_friction] no FrictionDRBamActuator on "
            f"'{asset_cfg.name}' — friction randomization is a NO-OP."
        )
