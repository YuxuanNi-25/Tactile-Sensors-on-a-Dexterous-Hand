#!/usr/bin/env python3
"""
Interactive Shadow Hand tactile simulation in PyBullet.

- Shadow Hand from dex-urdf.
- A dynamic plate (box) with non-zero mass in front of the hand:
  it moves under gravity and fingertip forces.
- Each finger is controlled by a GUI slider (0..1).
  * 0.0  = fully open (joint angle = 0 rad)
  * 1.0  = gently closed (small joint angle, only a few degrees)
- IMPORTANT:
  * No automatic motion: we do NOT use POSITION_CONTROL.
  * All joints are kinematic and only move when you change the slider.
  * This eliminates oscillations / "twitching".
- For each time step, we compute the net contact wrench at 5 fingertip links,
  in both world and palm frames.
- All frames are logged to 'wrench_log.npy'.
- Press Ctrl+C in the terminal to stop; the script will save whatever
  has been recorded so far and then disconnect from PyBullet.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data

# URDF path relative to dex-urdf/robots
SHADOW_URDF_REL_PATH = "hands/shadow_hand/shadow_hand_right.urdf"


def init_sim():
    """Initialize PyBullet, load Shadow Hand, and create a dynamic plate."""
    physics_client = p.connect(p.GUI)

    # Search path for plane.urdf, etc.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Search path for dex-urdf robots (Shadow hand)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    dex_robots_path = os.path.join(this_dir, "dex-urdf", "robots")
    p.setAdditionalSearchPath(dex_robots_path)

    # Basic physics configuration
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.setRealTimeSimulation(0)

    # Make contacts more stable
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=150,
        numSubSteps=10,
        solverResidualThreshold=1e-7,
        contactBreakingThreshold=0.0005,
        warmStartingFactor=0.1,
    )

    # Ground plane
    plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    ground_id = p.loadURDF(plane_path)

    # Shadow Hand base pose:
    # Rotate so that the palm roughly faces upward (+z) and fingers point forward (+x).
    hand_start_pos = [0, 0, 0.25]
    hand_start_orn = p.getQuaternionFromEuler([0, -1.57, 0])

    hand_id = p.loadURDF(
        SHADOW_URDF_REL_PATH,
        hand_start_pos,
        hand_start_orn,
        useFixedBase=True,
    )

    # Find palm link so we know where the palm is in world coordinates
    palm_link_index = find_palm_link_index(hand_id)
    if palm_link_index >= 0:
        palm_state = p.getLinkState(hand_id, palm_link_index)
        palm_pos, _ = palm_state[0], palm_state[1]
    else:
        palm_pos, _ = p.getBasePositionAndOrientation(hand_id)

    # -------------------------------------------------------------
    # Create a dynamic box ("plate") in front of the palm.
    #
    # Orientation: identity => XY plane, normal = +z (same as ground).
    # Offset it along +x (rough finger direction) so it does not
    # intersect the hand at spawn, and slightly above in +z so it can
    # fall under gravity.
    # -------------------------------------------------------------

    plate_orn = p.getQuaternionFromEuler([0, 0, 0])

    plate_x_offset = -0.08   # along finger direction (+x), move away from hand
    plate_y_offset = 0.03
    plate_z_offset = 0.05   # 3 cm above palm
    plate_pos = [
        palm_pos[0] + plate_x_offset,
        palm_pos[1] + plate_y_offset,
        palm_pos[2] + plate_z_offset,
    ]

    # Box half-extents: thin in z (thickness), large in x/y
    plate_half_extents = [0.08, 0.08, 0.01]  # larger plate

    plate_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=plate_half_extents,
    )
    plate_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=plate_half_extents,
        rgbaColor=[0.2, 0.2, 1.0, 0.5],  # semi-transparent blue
    )

    # Non-zero mass ⇒ dynamic body (affected by gravity and contact forces)
    plate_mass = 3  # 3 kg plate

    plate_id = p.createMultiBody(
        baseMass=plate_mass,
        baseCollisionShapeIndex=plate_collision,
        baseVisualShapeIndex=plate_visual,
        basePosition=plate_pos,
        baseOrientation=plate_orn,
    )

    # Dynamics settings
    p.changeDynamics(ground_id, -1, lateralFriction=1.0, restitution=0.0)

    num_joints = p.getNumJoints(hand_id)
    for j in range(-1, num_joints):
        p.changeDynamics(
            hand_id,
            j,
            lateralFriction=1.0,
            restitution=0.0,
            # Slightly inflate collision shapes to reduce tunneling
            collisionMargin=0.01,
        )

    p.changeDynamics(
        plate_id,
        -1,
        lateralFriction=0.8,
        restitution=0.0,
    )

    print("Initialized simulation with dynamic plate id:", plate_id)
    return hand_id, palm_link_index, plate_id, physics_client


def find_palm_link_index(body_id):
    """Try to find a link whose name contains 'palm'."""
    num_joints = p.getNumJoints(body_id)
    for j in range(num_joints):
        info = p.getJointInfo(body_id, j)
        link_name = info[12].decode("utf-8").lower()
        if "palm" in link_name:
            return j
    return -1


def build_finger_groups(body_id):
    """
    Group Shadow Hand joints into 5 fingers based on name patterns.

    Returns:
        dict: {finger_name: [joint_indices]}
    """
    num_joints = p.getNumJoints(body_id)

    group_patterns = {
        "thumb": ["THJ"],
        "index": ["FFJ"],
        "middle": ["MFJ"],
        "ring": ["RFJ"],
        "little": ["LFJ"],
    }

    finger_groups = {k: [] for k in group_patterns.keys()}

    for j in range(num_joints):
        info = p.getJointInfo(body_id, j)
        joint_name = info[1].decode("utf-8")
        for gname, patterns in group_patterns.items():
            if any(pat in joint_name for pat in patterns):
                finger_groups[gname].append(j)

    for gname in finger_groups:
        finger_groups[gname] = sorted(finger_groups[gname])

    print("Finger groups (joint indices):")
    for gname, joints in finger_groups.items():
        print(gname, ":", joints)

    return finger_groups


def get_all_finger_joint_indices(finger_groups):
    """Flatten all finger-group joint indices into a single list."""
    indices = []
    for joints in finger_groups.values():
        indices.extend(joints)
    return indices


def guess_fingertip_links_from_groups(body_id, finger_groups):
    """
    For each finger group, choose the joint with the largest index
    and take its child link as the "fingertip link".
    """
    fingertip_links = []

    num_joints = p.getNumJoints(body_id)
    for gname, joints in finger_groups.items():
        if not joints:
            continue
        j_distal = max(joints)
        link_idx = j_distal  # in PyBullet, joint j controls link j
        if 0 <= link_idx < num_joints:
            fingertip_links.append(link_idx)
            info = p.getJointInfo(body_id, j_distal)
            link_name = info[12].decode("utf-8")
            print(f"Finger '{gname}' distal joint {j_distal}, link {link_idx}, name={link_name}")
        else:
            print(f"Finger '{gname}': distal joint {j_distal} invalid link index {link_idx}")

    print("Auto-selected fingertip link indices:", fingertip_links)
    return fingertip_links


def build_small_close_targets(body_id, joint_indices, max_abs_angle=0.15):
    """
    Build per-joint closing targets limited to a VERY small range.

    This keeps finger motion very gentle:
    - max_abs_angle ~ 0.15 rad ~= 8.6 degrees.
    - Slider = 1.0 => at most ~8.6 deg of joint motion.
    """
    joint_targets = {}
    for j in joint_indices:
        info = p.getJointInfo(body_id, j)
        lo = info[8]
        hi = info[9]

        # If limits are huge, treat it as continuous and clamp to +/- max_abs_angle
        if hi - lo > 1000:
            hi = max_abs_angle
            lo = -max_abs_angle

        if hi > 0:
            raw_target = min(hi, max_abs_angle)
        elif hi < 0:
            raw_target = max(hi, -max_abs_angle)
        else:
            raw_target = max_abs_angle * 0.5

        target = max(lo, min(hi, raw_target))
        joint_targets[j] = target

    print("Per-joint closing targets (rad, very small):")
    for j in joint_indices:
        print(f"  joint {j}: {joint_targets[j]:.3f}")
    return joint_targets


def compute_link_wrench_world(body_id, link_index):
    """
    Compute the net 6D contact wrench (force and torque) at a given link
    in WORLD coordinates.
    """
    link_state = p.getLinkState(body_id, link_index)
    link_pos_world = np.array(link_state[0])

    cps = p.getContactPoints(bodyA=body_id)

    F_world = np.zeros(3)
    tau_world = np.zeros(3)

    for cp in cps:
        bodyA = cp[1]
        bodyB = cp[2]
        linkA = cp[3]
        linkB = cp[4]

        if bodyA == body_id and linkA == link_index:
            pos_on_hand = np.array(cp[5])      # positionOnA in world
            normal_on_B = np.array(cp[7])      # normal on B, from B to A
            normal_force = cp[9]
            # Force on A is opposite of force on B
            F_contact = -normal_force * normal_on_B
        elif bodyB == body_id and linkB == link_index:
            pos_on_hand = np.array(cp[6])      # positionOnB in world
            normal_on_B = np.array(cp[7])      # normal on B, from B to A
            normal_force = cp[9]
            # Force on B is along normal_on_B
            F_contact = normal_force * normal_on_B
        else:
            continue

        F_world += F_contact
        r = pos_on_hand - link_pos_world
        tau_world += np.cross(r, F_contact)

    return F_world, tau_world


def world_to_palm_force_tau(F_world, tau_world, hand_id, palm_link_index):
    """
    Transform force and torque from WORLD frame to PALM frame.
    """
    palm_state = p.getLinkState(hand_id, palm_link_index)
    palm_orn_world = palm_state[1]

    R_palm_to_world = np.array(p.getMatrixFromQuaternion(palm_orn_world)).reshape(3, 3)
    R_world_to_palm = R_palm_to_world.T

    F_palm = R_world_to_palm @ F_world
    tau_palm = R_world_to_palm @ tau_world

    return F_palm, tau_palm


def visualize_force_arrow_world(body_id, link_index, F_world, life_time=0.05):
    """
    Draw a line in the GUI corresponding to the net WORLD-frame force at a link.
    """
    link_state = p.getLinkState(body_id, link_index)
    link_pos_world = link_state[0]

    start = link_pos_world
    scale = 1.0 / 5.0  # visualization scale (bigger for small forces)
    end = [
        start[0] + F_world[0] * scale,
        start[1] + F_world[1] * scale,
        start[2] + F_world[2] * scale,
    ]

    p.addUserDebugLine(start, end, lineWidth=2, lifeTime=life_time)


if __name__ == "__main__":
    hand_id, palm_link_index, plate_id, client = init_sim()

    # Build finger groups and joint lists
    finger_groups = build_finger_groups(hand_id)
    all_finger_joints = get_all_finger_joint_indices(finger_groups)

    # Auto-select fingertip links from finger groups
    FINGERTIP_LINK_INDICES = guess_fingertip_links_from_groups(hand_id, finger_groups)
    if not FINGERTIP_LINK_INDICES:
        FINGERTIP_LINK_INDICES = [10, 13, 16, 19, 22]
        print("Warning: using fallback fingertip link indices:", FINGERTIP_LINK_INDICES)

    # Per-joint gentle (VERY SMALL) closing targets
    joint_close_targets = build_small_close_targets(
        hand_id, all_finger_joints, max_abs_angle=0.15
    )

    # === Phase 1: open hand and let plate fall & settle in front of the hand ===
    # Set all joints to 0 rad (open and static)
    for j in all_finger_joints:
        p.resetJointState(hand_id, j, 0.0)

    settle_steps = 600  # let plate settle under gravity
    for _ in range(settle_steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # === Create GUI sliders for each finger ===
    #
    # Slider range: 0.0 (fully open) to 1.0 (reach very small joint_close_targets).
    finger_slider_ids = {}
    for gname in finger_groups.keys():
        slider_id = p.addUserDebugParameter(
            f"close_{gname}", 0.0, 1.0, 0.0
        )
        finger_slider_ids[gname] = slider_id

    print("Use the sliders in the PyBullet GUI to control each finger.")
    print("Sliders are deliberately 'low gain' (hand moves very little).")
    print("Press Ctrl+C in this terminal to stop and save the log.")

    logged_wrenches = []

    try:
        while True:
            # Read slider values (per finger: 0..1)
            alpha_per_finger = {}
            for gname, slider_id in finger_slider_ids.items():
                alpha_per_finger[gname] = p.readUserDebugParameter(slider_id)

            # Apply joint angles *kinematically* using resetJointState.
            # No motors, no automatic motion: only what you set via sliders.
            for gname, joints in finger_groups.items():
                alpha = alpha_per_finger.get(gname, 0.0)
                for j in joints:
                    target_angle = alpha * joint_close_targets[j]
                    p.resetJointState(hand_id, j, target_angle)

            # Step simulation: only the plate and other dynamic bodies move.
            p.stepSimulation()

            # Log fingertip wrenches
            frame_data = {}
            for link_idx in FINGERTIP_LINK_INDICES:
                F_world, tau_world = compute_link_wrench_world(hand_id, link_idx)
                F_palm, tau_palm = world_to_palm_force_tau(
                    F_world, tau_world, hand_id, palm_link_index
                )

                frame_data[link_idx] = {
                    "F_world": F_world,
                    "tau_world": tau_world,
                    "F_palm": F_palm,
                    "tau_palm": tau_palm,
                }

                visualize_force_arrow_world(hand_id, link_idx, F_world)

            logged_wrenches.append(frame_data)

            # Run close to real-time so you can see what you're doing
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\n[sim] Interrupted by user (Ctrl+C). Saving partial log...")

    finally:
        np.save("wrench_log.npy", logged_wrenches, allow_pickle=True)
        print(f"[sim] Saved {len(logged_wrenches)} frames to wrench_log.npy")
        p.disconnect()
