# Copyright 2026 ftservo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-episode friction domain randomization and encoder-through-backlash
feedback for the plain MuJoCo pipeline.

Companion of :mod:`bam.mujoco`, and the pure-MuJoCo counterpart of
:mod:`bam.feetech.mjlab_friction_dr`, which does the same job for mjlab /
MuJoCo-Warp. The two are deliberately separate modules rather than one module
with an optional extra: they share the *idea* but not a single line of the
lifecycle they hook into.


Where this lives
----------------
``bam/feetech/mujoco_friction_dr.py`` — a sibling of
:mod:`bam.feetech.mjlab_friction_dr`, so the two pipelines' randomization code
sits together. It is **not** re-exported from ``bam/feetech/__init__.py`` (which
stays empty on purpose), so nothing is loaded unless asked for::

    from bam.feetech.mujoco_friction_dr import FrictionDRMujocoController

Being under ``bam/feetech/`` is a *dependency* choice as well as a conventional
one: the module itself subclasses :class:`~bam.mujoco.MujocoController` (one level
up) and is servo-family agnostic, but ``scripts/verify_mujoco_friction_dr.py``
looks for it **only in the driver directory of the servo under test** — the
directory holding the ``actuator.py`` that defines that servo, read out of
``bam/actuators.py`` without importing it (``--json-path sts3215`` →
``bam/feetech``, ``xl330`` → ``bam/dynamixel``). An ``xl330`` run therefore
aborts here rather than silently verifying this file, and covering another family
means putting a copy of the module beside that family's driver — sibling driver
directories sit at the same depth, so the relative imports below stay valid
as-is.

Nothing here imports the GPU stack: numpy + mujoco only, no torch and no
``[mjlab]`` extra, so it runs anywhere :mod:`bam.mujoco` runs.


bam/mujoco.py is left untouched
-------------------------------
Everything below is subclassing and composition; ``git diff bam/mujoco.py``
stays empty. That is on purpose — the file is upstream code that gets replaced
wholesale when the package is updated (1.0.1 → 1.0.2 already changed
``MujocoController.reset`` in a breaking way, and this repo carries a second,
vendored copy), so a local patch there is the first thing to be lost.

The one thing that genuinely needs a new seam is
:meth:`~bam.mujoco.MujocoController.update`, which reads the joint position it
feeds to the control law as ``self.mujoco_data.qpos[self.qpos_indexes]`` and
offers no way to substitute it. Rather than re-implement that ~90-line method,
:class:`BacklashEncoderMujocoController` **hands the parent a different
``mujoco_data`` for the duration of the call**: a :class:`_EncoderQposView` that
forwards every attribute to the real ``MjData`` except ``qpos``, which it
replaces with the encoder's view of the joint state. ``super().update()`` then
runs unmodified — same solver fields, same friction budget, same code path — and
``self.mujoco_data`` is restored in a ``finally``, so the real ``MjData`` is what
callers see outside ``update()`` and no write can ever be lost to the view.

If a contributed ``_sensed_position()`` hook ever lands upstream, the view can be
deleted and the subclass reduced to a one-method override; nothing else changes.


How this differs from the mjlab module
--------------------------------------
* **What an "env" is.** mjlab batches N worlds inside one ``MjModel``. Plain
  MuJoCo has one ``MjModel``/``MjData`` pair per simulation, so the unit of
  randomization here is the *controller*: one controller per robot instance,
  each carrying its own ``friction_scale``. Vectorize the naive way — one
  ``(MjModel, MjData, controller)`` triplet per environment, exactly as
  :meth:`bam.mujoco.Simulator.reset` builds them. Inside a single controller,
  ``ids`` therefore index the *controlled actuators* (its ``actuator`` list),
  not environments.
* **Firmware state needs no deferral.** mjlab's ``reset()`` runs before the
  joint state is readable, so the stateful control-law buffer has to be re-seeded
  later, inside ``compute()``. Here ``reset()`` runs between steps with ``qpos``
  already valid, so the buffer is simply dropped and ``compute_control`` re-seeds
  it from the current position on the next call.
* **``load_config`` has a twin.** Upstream ``load_config`` hard-codes
  :class:`~bam.mujoco.MujocoController`, so the config-file workflow gets
  :func:`load_friction_dr_config` here instead of a hook upstream.


What this module adds — and what it does NOT need to add
--------------------------------------------------------
1. **Per-episode friction re-sampling.** The startup-only
   ``friction_scale_range`` gives a fixed population of N virtual robots, one
   scale each for the whole run. Standard domain randomization re-draws per
   episode; ``per_episode_friction_scale_range`` does that on every
   :meth:`FrictionDRMujocoController.reset`, and :func:`randomize_friction` does
   it from an explicit range when the range has to live in the task loop.

2. **Encoder reads through the gear backlash.**
   :class:`BacklashEncoderMujocoController` feeds the firmware position loop
   ``qpos[servo] + qpos[backlash]`` instead of ``qpos[servo]``, reproducing a
   servo whose magnetic encoder sits on the output side of the play: while the
   servo winds through the dead zone the measured position — and hence the PD
   error — does not change.

3. **Firmware-state reset on** :meth:`~FrictionDRMujocoController.reset` (works
   around an upstream defect, see
   :meth:`FrictionDRMujocoController._reset_firmware_state`).

.. note::
   :meth:`bam.mujoco.MujocoController.reset` already clears the state **it**
   owns (``control``, ``actuator_state``, ``last_ts``), and when an actuator
   declares ``stateful = True`` ``MujocoController`` already saves and restores
   its state around every ``compute_control`` call — the isolation contract is
   implemented on this side of the code base. A *stateful* actuator therefore
   needs nothing from this module beyond the friction randomization. (3) exists
   because ``stateful`` is ``False`` package-wide today, so
   ``STS3215Actuator.q_target_smooth`` — which *is* control-law state — survives
   a reset, exactly as it does under mjlab.

.. note::
   Verified against a real MuJoCo model and a real servo's fitted params by
   ``scripts/verify_mujoco_friction_dr.py``: which servo is exercised is a
   required command-line argument there (``--json-path``, resolved from the
   bundled params or any path), and ``--model-xml`` additionally checks the
   ``add_backlash.py`` contract on a robot MJCF.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable, Mapping

import mujoco
import numpy as np

from ..model import load_model_from_dict
from ..mujoco import MujocoController

if TYPE_CHECKING:
    from ..model import Model

__all__ = [
    "FrictionDRMujocoController",
    "BacklashEncoderMujocoController",
    "randomize_friction",
    "load_friction_dr_config",
]


def _check_scale_range(name: str, scale_range) -> tuple[float, float] | None:
    """Validate a friction-scale range and return it as a float pair.

    Rejected eagerly, at construction, rather than at the first reset: a range
    that is inverted, non-finite or non-positive would otherwise turn into
    ``nan`` friction (which MuJoCo silently accepts) or into a negative
    ``dof_frictionloss`` (which is physically meaningless).
    """
    if scale_range is None:
        return None
    lo, hi = (float(v) for v in scale_range)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} must be finite, got {scale_range!r}")
    if lo <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive (a friction multiplier of 0 would "
            f"erase friction entirely and a negative one is unphysical), got {lo}"
        )
    if hi < lo:
        raise ValueError(f"{name} is inverted: {scale_range!r}")
    return lo, hi


def _as_ids(ids, n: int) -> np.ndarray:
    """Normalize the accepted ``ids`` forms to an integer array.

    Accepts ``None``/``...`` (all of them), a slice, a list, an integer, a numpy
    array or a boolean mask — the same set the mjlab module's
    ``_env_ids_tensor`` takes.
    """
    if ids is None or ids is Ellipsis:
        return np.arange(n, dtype=int)
    if isinstance(ids, slice):
        return np.arange(n, dtype=int)[ids]
    arr = np.asarray(ids)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    return arr.astype(int).ravel()


class _EncoderQposView:
    """``MjData`` stand-in whose ``qpos`` is the encoder's view of the joints.

    Only ``qpos`` is substituted; every other attribute (``qvel``, ``time``,
    ``qfrc_*``, ``efc_*``, …) is forwarded to the wrapped ``MjData``, so a parent
    method that reads its state through ``self.mujoco_data`` behaves identically
    except at the one read that was meant to change.

    The substituted array is a **copy**, so writes through it would be lost.
    That is why the substitution is scoped to a single ``update()`` call and
    undone in a ``finally`` — see the module docstring.
    """

    __slots__ = ("_real", "_qpos")

    def __init__(self, real: mujoco.MjData, qpos: np.ndarray):
        self._real = real
        self._qpos = qpos

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class FrictionDRMujocoController(MujocoController):
    """``MujocoController`` with a per-actuator, re-samplable friction budget scale.

    The scale multiplies BAM's **entire** velocity-independent friction budget
    (Coulomb + Stribeck + load-dependent), which is the term carrying the
    dominant sim2real friction uncertainty (stiction / gearbox). The viscous
    (velocity-proportional) term still goes to ``dof_damping`` untouched — scale
    it too by overriding :meth:`update` if ever needed.

    The multiplication is applied on top of the budget
    :meth:`~bam.mujoco.MujocoController.update` has just written into
    ``dof_frictionloss``, i.e. still **before MuJoCo solves**, so the constraint
    solver keeps doing the static-friction clipping (BAM Algorithm 1). Adding a
    passive friction torque to the commanded motor torque instead would bypass
    that clipping. Because ``compute_frictions`` recomputes the unscaled budget
    on every call, the scaling never accumulates: calling :meth:`update` twice
    per timestep gives the same result as calling it once.

    :param friction_scale_range: Startup-only per-actuator scale, drawn once in
        the constructor (like mass DR). ``None`` (default) → all ones.
    :param per_episode_friction_scale_range: If set, a fresh scale is drawn from
        this range on **every** :meth:`reset`, e.g. ``(0.9, 1.1)``. Configure
        this *or* ``friction_scale_range`` (startup-only) — if both are given,
        the startup draw defines the baseline that :meth:`reset_friction_scale`
        restores and each episode overwrites it.
    :param reset_firmware_state: Drop the control law's own state on
        :meth:`reset`, so it re-seeds from the current position instead of
        slewing in from the previous episode. Harmless for actuators holding no
        such state.
    :param rng: ``numpy.random.Generator`` used for every draw. Pass a seeded
        one for reproducible runs; defaults to an independent fresh generator.
    """

    def __init__(
        self,
        model: "Model",
        actuator,
        mujoco_model: mujoco.MjModel,
        mujoco_data: mujoco.MjData,
        vin_drop_resistance: float | None = None,
        vin_min: float | None = None,
        friction_scale_range: tuple[float, float] | None = None,
        per_episode_friction_scale_range: tuple[float, float] | None = None,
        reset_firmware_state: bool = True,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            model,
            actuator,
            mujoco_model,
            mujoco_data,
            vin_drop_resistance,
            vin_min,
        )
        self.per_episode_friction_scale_range = _check_scale_range(
            "per_episode_friction_scale_range", per_episode_friction_scale_range
        )
        self.reset_firmware_state = reset_firmware_state
        self._rng = rng if rng is not None else np.random.default_rng()

        startup = _check_scale_range("friction_scale_range", friction_scale_range)
        n = len(self.actuator)
        self.friction_scale = np.ones(n, dtype=float)
        if startup is not None:
            self.friction_scale = self._rng.uniform(startup[0], startup[1], size=n)
        # Baseline restored by reset_friction_scale, so per-episode sampling can
        # never accumulate across resets.
        self.default_friction_scale = self.friction_scale.copy()

        print(
            f"[FrictionDRMujocoController] actuators={list(np.atleast_1d(actuator))} "
            f"dofs={np.asarray(self.dof_indexes).tolist()} "
            f"startup_scale_range={startup} "
            f"per_episode_scale_range={self.per_episode_friction_scale_range} "
            f"reset_firmware_state={self.reset_firmware_state}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-actuator friction scale API
    # ─────────────────────────────────────────────────────────────────────────

    def _actuator_ids(self, ids) -> np.ndarray:
        """Normalize ``ids`` against this controller's actuator list."""
        return _as_ids(ids, len(self.actuator))

    def set_friction_scale(self, ids, friction_scale) -> None:
        """Write an explicit friction scale for the selected actuators.

        ``friction_scale`` is broadcast against the selection, so a scalar or a
        ``(len(ids),)`` array both work.

        .. important::
            Call :meth:`reset_friction_scale` first (or let
            ``per_episode_friction_scale_range`` do it) — the scale is a
            *multiplier*, so a fresh sample must start from the 1.0 baseline
            rather than from the previous episode's value.
        """
        sel = self._actuator_ids(ids)
        values = np.broadcast_to(np.asarray(friction_scale, dtype=float), (len(sel),))
        self.friction_scale[sel] = values

    def reset_friction_scale(self, ids=None) -> None:
        """Restore the startup baseline scale for the selected actuators.

        The baseline is all ones when ``friction_scale_range`` was not given, and
        the value drawn at construction when it was.
        """
        sel = self._actuator_ids(ids)
        self.friction_scale[sel] = self.default_friction_scale[sel]

    def sample_friction_scale(self, ids=None, scale_range=None) -> np.ndarray:
        """Re-draw the scale and return the ``(len(ids),)`` sample.

        The range comes from ``scale_range`` when given, otherwise from
        ``per_episode_friction_scale_range``; with neither set this raises rather
        than silently doing nothing. Non-accumulating: the baseline is restored
        before writing, so calling it a thousand times stays inside the range.
        """
        source = (
            scale_range
            if scale_range is not None
            else self.per_episode_friction_scale_range
        )
        if source is None:
            raise ValueError(
                "no scale range to sample from: pass scale_range=..., or set "
                "per_episode_friction_scale_range at construction"
            )
        lo, hi = _check_scale_range("scale_range", source)
        sel = self._actuator_ids(ids)
        samples = self._rng.uniform(lo, hi, size=len(sel))
        self.reset_friction_scale(sel)
        self.set_friction_scale(sel, samples)
        return samples

    # ─────────────────────────────────────────────────────────────────────────
    # Firmware-state reset (upstream defect workaround)
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_firmware_state(self) -> None:
        """Drop the control law's state so it re-seeds from the current position.

        ``bam.actuator.Actuator.reset`` is a bare ``pass`` and ``stateful`` is
        ``False`` package-wide, so nothing upstream reaches
        ``STS3215Actuator.q_target_smooth`` — which *is* control-law state.
        Left alone, an episode that teleports the joint starts with the
        rate-limited internal goal slewing in from the previous episode's value:
        with the 1.0.2 ``max_velocity`` default of 7.0 rad/s and ``dt = 5 ms``
        that is 0.035 rad per step, ~29 steps to recover a 1 rad teleport.

        Setting it to ``None`` is precisely the "not initialised yet" state
        ``compute_control`` checks for, so the next call seeds it from the
        current position. No-op for actuators without such a buffer (HD1910,
        XL330, …), which is why it is safe to leave enabled unconditionally.

        .. note::
            The recommended *upstream* fix is to declare ``stateful = True`` on
            ``STS3215Actuator`` and implement ``get_state``/``set_state``, which
            is exactly the contract :meth:`bam.mujoco.MujocoController.update`
            already honours. Until then, note that with ``stateful = False`` the
            buffer lives on the shared :class:`~bam.model.Model`, so this cannot
            be done *per instance*: two controllers built on one ``Model`` share
            it, and dropping it for one drops it for both. Give each instance its
            own ``Model`` (as :meth:`bam.mujoco.Simulator.reset` does) if that
            matters.
        """
        if getattr(self.model.actuator, "q_target_smooth", None) is not None:
            self.model.actuator.q_target_smooth = None

    # ─────────────────────────────────────────────────────────────────────────
    # MuJoCo lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, qpos=None) -> None:  # noqa: ANN001
        """Episode reset: restore the target, drop the firmware state, re-sample.

        Call it after resetting the simulation state (``mj_resetData``, a
        teleport, a new episode), *before* the next :meth:`update`.

        :param qpos: Optional full ``mj_data.qpos`` array. When given, the target
            is set to the controlled joints' positions in it — the 1.0.1
            ``reset(qpos)`` calling convention, which is also what
            ``scripts/testbench_sim2real.py`` uses. When omitted the target is
            left alone, matching ``reset()`` in 1.0.2, so a caller that has
            already called :meth:`set_q_target` keeps its goal.
        """
        super().reset()
        if qpos is not None:
            self.q_target = np.asarray(qpos, dtype=float)[self.qpos_indexes]
        if self.reset_firmware_state:
            self._reset_firmware_state()
        if self.per_episode_friction_scale_range is not None:
            self.sample_friction_scale()

    def _encoder_qpos(self) -> np.ndarray | None:
        """``qpos`` as the firmware's encoder sees it, or ``None`` for the plain one.

        The seam for encoder placement: returning an array here makes
        :meth:`update` hand the parent control law that view of the joint state
        instead of the real ``qpos``, without re-implementing ``update``. The
        default is ``None`` — no copy, no view, the fast path.
        """
        return None

    def update(self):
        """Run the plain controller, then apply the friction scale.

        See the class docstring for why the multiplication sits here — after the
        budget is written, before MuJoCo solves. When :meth:`_encoder_qpos`
        returns a view, the parent call runs against a :class:`_EncoderQposView`
        that is swapped in for the duration of the call and restored afterwards.
        """
        encoder_qpos = self._encoder_qpos()
        if encoder_qpos is None:
            super().update()
        else:
            real_data = self.mujoco_data
            self.mujoco_data = _EncoderQposView(real_data, encoder_qpos)
            try:
                super().update()
            finally:
                self.mujoco_data = real_data
        self.mujoco_model.dof_frictionloss[self.dof_indexes] *= self.friction_scale


class BacklashEncoderMujocoController(FrictionDRMujocoController):
    """``FrictionDRMujocoController`` whose firmware PD reads the encoder THROUGH backlash.

    Backlash models put an unactuated ``passive_<joint>_backlash`` hinge in
    series with each servo joint: the servo joint is the motor output, the
    backlash joint is the play between it and the link, and the link angle is
    their sum.

    On the real servo the magnetic encoder sits on the OUTPUT side of that play,
    so the firmware position loop closes on ``main + backlash`` — while the servo
    winds through the dead zone the measured position (and hence the PD error)
    does not change. This subclass reproduces that through the
    :meth:`~FrictionDRMujocoController._encoder_qpos` seam.

    Velocity is left motor-side on purpose: in BAM ``dq`` drives back-EMF and
    friction, which are rotor physics, not an encoder-derived firmware signal.

    The play joints are resolved **by name** when the controller is built (the
    MuJoCo model is already compiled and immutable at this point, so unlike the
    mjlab version there is nothing to defer). A joint without a play companion
    keeps ``mask = 0`` and is sensed normally, so the class degrades to a plain
    friction-DR controller on any robot model — including mixed models where only
    some joints have play.

    :param backlash_joint_pattern: Name template for the play hinge. The default
        is the ``add_backlash.py`` convention; override it for another naming
        scheme.

    .. note::
        The ``+`` above assumes the play hinge is on the **same body and the same
        axis direction** as the motor joint it follows (the contract
        ``add_backlash.py`` implements). A flipped or rotated axis would be
        silently wrong here, so pass the robot's MJCF to
        ``scripts/verify_mujoco_friction_dr.py --model-xml`` (or to the mjlab
        harness) to have that checked per joint, along with full coverage — no
        hinge left without play.

    .. warning::
        Pair this with matching **observations**. If the policy observes the
        motor-side angle while the firmware closes on the encoder view, the
        policy is trained against a channel the real robot does not provide.
        Both must read :meth:`encoder_position`. Also exclude the
        ``passive_*_backlash`` joints from any soft-limit penalty reward (they
        legitimately ride their hard limits) and from regex-based joint
        selections that would otherwise match them ambiguously.
    """

    def __init__(
        self,
        *args,
        backlash_joint_pattern: str = "passive_{joint}_backlash",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backlash_joint_pattern = backlash_joint_pattern
        self._resolve_backlash_joints()
        n_play = int(self._backlash_mask.sum())
        print(
            f"[BacklashEncoderMujocoController] encoder-through-backlash feedback "
            f"on {n_play}/{len(self.actuator)} joints"
        )
        if n_play == 0:
            print(
                "[BacklashEncoderMujocoController] no "
                f"'{backlash_joint_pattern}' joints found — behaving as a plain "
                "FrictionDRMujocoController."
            )

    def _resolve_backlash_joints(self) -> None:
        """Look up each controlled joint's play hinge by name.

        Stores, per controlled actuator, the play joint's ``qpos`` address (index
        ``0`` as a harmless placeholder when there is no play joint) and a
        ``0/1`` mask selecting it, mirroring the mjlab implementation.
        """
        n = len(self.actuator)
        self.backlash_joint_names: list[str | None] = []
        self.backlash_joint_ids = np.full(n, -1, dtype=int)
        qpos_addr = np.zeros(n, dtype=int)
        mask = np.zeros(n, dtype=float)
        for i in range(n):
            main = mujoco.mj_id2name(
                self.mujoco_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.joint_indexes[i]),
            )
            name = (
                None if main is None else self.backlash_joint_pattern.format(joint=main)
            )
            self.backlash_joint_names.append(name)
            if name is None:
                continue
            jid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                self.backlash_joint_names[i] = None
                continue
            self.backlash_joint_ids[i] = jid
            qpos_addr[i] = int(self.mujoco_model.jnt_qposadr[jid])
            mask[i] = 1.0
        if not len(self.mujoco_data.qpos):
            raise RuntimeError("model has no qpos — cannot resolve backlash joints")
        # A missing play joint must not be read at all; index 0 with mask 0 does
        # that without a branch in the hot path.
        self._backlash_qpos_addresses = qpos_addr
        self._backlash_mask = mask

    def encoder_position(self) -> np.ndarray:
        """The joint angle(s) the firmware closes its loop on [rad].

        Also the value a matching observation must use — see the class warning.
        """
        return self.mujoco_data.qpos[self.qpos_indexes] + self.mujoco_data.qpos[
            self._backlash_qpos_addresses
        ] * self._backlash_mask

    def _encoder_qpos(self) -> np.ndarray | None:
        """A copy of ``qpos`` with ``main + backlash`` written into the servo joints."""
        if not self._backlash_mask.any():
            return None
        qpos = np.array(self.mujoco_data.qpos, dtype=float)
        qpos[self.qpos_indexes] += (
            self.mujoco_data.qpos[self._backlash_qpos_addresses] * self._backlash_mask
        )
        return qpos


def randomize_friction(
    controllers: "FrictionDRMujocoController | Iterable | Mapping",
    scale_range: tuple[float, float],
    ids=None,
) -> int:
    """Episode-reset hook: re-sample the friction scale of every DR controller.

    Use it when the range must live in the *training loop* rather than in the
    controller — e.g. a curriculum that widens ``scale_range`` over episodes.
    Call it right after resetting the simulation, so the new scale is in place
    before the first :meth:`~FrictionDRMujocoController.update` of the episode.

    :param controllers: one controller, or any iterable/mapping of them (the dict
        :func:`bam.mujoco.load_config` returns works directly). Entries that are
        not friction-DR controllers are skipped, so it is safe to pass a mixed
        collection.
    :param scale_range: ``(lo, hi)`` to draw from; both must be positive.
    :param ids: Restrict the re-draw to some of each controller's actuators
        (same forms as :meth:`FrictionDRMujocoController.set_friction_scale`).
    :returns: How many controllers were actually re-sampled. ``0`` means the call
        was a no-op — the same failure mode as forgetting the hook entirely, so
        it prints a warning rather than returning quietly.
    """
    if isinstance(controllers, FrictionDRMujocoController):
        items = [controllers]
    elif isinstance(controllers, Mapping):
        items = list(controllers.values())
    else:
        items = list(controllers)

    matched = 0
    for controller in items:
        if isinstance(controller, FrictionDRMujocoController):
            controller.sample_friction_scale(ids, scale_range=scale_range)
            matched += 1

    if matched == 0:
        print(
            "[randomize_friction] no FrictionDRMujocoController among the given "
            "controllers — friction randomization is a NO-OP."
        )
    return matched


def load_friction_dr_config(
    path: str,
    mujoco_model: mujoco.MjModel,
    mujoco_data: mujoco.MjData,
    kp: float,
    vin: float,
    controller_class=FrictionDRMujocoController,
    **controller_kwargs,
) -> tuple:
    """Friction-DR twin of :func:`bam.mujoco.load_config`.

    Same config file, same mapping dicts, but the controllers it returns are
    friction-DR capable. It exists because ``load_config`` hard-codes
    :class:`~bam.mujoco.MujocoController`; keeping this module free of upstream
    edits means re-implementing the 20-line loader rather than adding a
    ``controller_class`` hook there.

    :param path: path to the configuration file (as for ``load_config``).
    :param mujoco_model: the mujoco model.
    :param mujoco_data: the mujoco data.
    :param kp: the proportional gain.
    :param vin: the input voltage.
    :param controller_class: controller class to instantiate, defaults to
        :class:`FrictionDRMujocoController`; pass
        :class:`BacklashEncoderMujocoController` for the encoder-through-play
        variant.
    :param controller_kwargs: extra keyword arguments forwarded to
        ``controller_class`` (e.g.
        ``per_episode_friction_scale_range=(0.9, 1.1)``).

    :returns: Tuple ``(controllers, dof_to_controller)`` — a dict keyed like the
        config file, and the dof → controller-key mapping, exactly as
        ``load_config`` returns them.
    """
    controllers = {}
    dof_to_controller = {}
    with open(path) as f:
        config = json.load(f)
    for key, value in config.items():
        dofs = value["dofs"]
        for dof in dofs:
            dof_to_controller[dof] = key

        model = load_model_from_dict(value["model"])
        model.actuator.kp = kp
        model.actuator.vin = vin
        model.actuator.error_gain = value["error_gain"]
        model.actuator.max_pwm = value["max_pwm"]

        controllers[key] = controller_class(
            model, dofs, mujoco_model, mujoco_data, **controller_kwargs
        )
        controllers[key].dofs = dofs

    return controllers, dof_to_controller
