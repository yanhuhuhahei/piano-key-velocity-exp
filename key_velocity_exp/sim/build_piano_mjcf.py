"""Programmatically build the 88-key piano MJCF model (RoboPianist geometry: hinged box keys with a
spring, 10 mm white-key dip at the tip). Used by the rigid-ball validation (kve-sim-ball)."""

import math

from dm_control import mjcf

from key_velocity_exp.sim import piano_constants as consts

WHITE_KEY_INDICES = consts.WHITE_KEY_INDICES
BLACK_TWIN_KEY_INDICES = consts.BLACK_TWIN_KEY_INDICES
BLACK_TRIPLET_KEY_INDICES = consts.BLACK_TRIPLET_KEY_INDICES


def build() -> mjcf.RootElement:
    """Programmatically build a piano MJCF."""
    root = mjcf.RootElement()
    root.model = "piano"

    root.compiler.autolimits = True
    root.compiler.angle = "radian"

    # Add materials.
    root.asset.add("material", name="white", rgba=consts.WHITE_KEY_COLOR)
    root.asset.add("material", name="black", rgba=consts.BLACK_KEY_COLOR)

    root.default.geom.type = "box"
    root.default.geom.solref = [0.004, 1]
    root.default.joint.type = "hinge"
    root.default.joint.axis = [0, 1, 0]
    root.default.site.type = "box"
    root.default.site.group = 5
    root.default.site.rgba = [1, 0, 0, 1]

    # This effectively disables key-key collisions but still allows hand-key collisions,
    # assuming we've kept the default hand contype = conaffinity = 1.
    # See https://mujoco.readthedocs.io/en/latest/computation.html#selection for more
    # details.
    root.default.geom.contype = 0
    root.default.geom.conaffinity = 1

    # White key defaults.
    white_default = root.default.add("default", dclass="white_key")
    white_default.geom.material = "white"
    white_default.geom.size = [
        consts.WHITE_KEY_LENGTH / 2,
        consts.WHITE_KEY_WIDTH / 2,
        consts.WHITE_KEY_HEIGHT / 2,
    ]
    white_default.geom.mass = consts.WHITE_KEY_MASS
    # white_default.geom.group = 3
    white_default.site.size = white_default.geom.size
    white_default.joint.pos = [-consts.WHITE_KEY_LENGTH / 2, 0, 0]
    white_default.joint.damping = consts.WHITE_JOINT_DAMPING
    white_default.joint.armature = consts.WHITE_JOINT_ARMATURE
    white_default.joint.stiffness = consts.WHITE_KEY_STIFFNESS
    white_default.joint.springref = consts.WHITE_KEY_SPRINGREF * math.pi / 180
    white_default.joint.range = [0, consts.WHITE_KEY_JOINT_MAX_ANGLE]

    # Black key defaults.
    black_default = root.default.add("default", dclass="black_key")
    black_default.geom.material = "black"
    black_default.geom.size = [
        consts.BLACK_KEY_LENGTH / 2,
        consts.BLACK_KEY_WIDTH / 2,
        consts.BLACK_KEY_HEIGHT / 2,
    ]
    black_default.geom.mass = consts.BLACK_KEY_MASS
    # black_default.geom.group = 3
    black_default.site.size = black_default.geom.size
    black_default.joint.pos = [-consts.BLACK_KEY_LENGTH / 2, 0, 0]
    black_default.joint.damping = consts.BLACK_JOINT_DAMPING
    black_default.joint.armature = consts.BLACK_JOINT_ARMATURE
    black_default.joint.stiffness = consts.BLACK_KEY_STIFFNESS
    black_default.joint.springref = consts.BLACK_KEY_SPRINGREF * math.pi / 180
    black_default.joint.range = [0, consts.BLACK_KEY_JOINT_MAX_ANGLE]

    piano_body = root.worldbody.add("body", name="piano_body", pos=[0, 0, 0])
    # Add base.
    base_body = piano_body.add("body", name="base", pos=consts.BASE_POS)
    base_body.add("geom", type="box", name="base_geom", size=consts.BASE_SIZE, rgba=consts.BASE_COLOR)
    base_body.add("joint", type="slide", name="base_joint", damping=100.0)

    # These will hold kwargs. We'll subsequently use them to create the actual objects.
    geoms = []
    bodies = []
    joints = []
    sites = []

    for i in range(consts.NUM_WHITE_KEYS):
        y_coord = -consts.PIANO_LENGTH * 0.5 + consts.WHITE_KEY_WIDTH * 0.5 + i * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
        bodies.append(
            {
                "name": f"white_key_{WHITE_KEY_INDICES[i]}",
                "pos": [consts.WHITE_KEY_X_OFFSET, y_coord, consts.WHITE_KEY_Z_OFFSET],
            }
        )
        geoms.append(
            {
                "name": f"white_key_geom_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )
        joints.append(
            {
                "name": f"white_joint_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )
        sites.append(
            {
                "name": f"white_key_site_{WHITE_KEY_INDICES[i]}",
                "dclass": "white_key",
            }
        )

    # Place the lone black key on the far left.
    y_coord = consts.WHITE_KEY_WIDTH + 0.5 * (-consts.PIANO_LENGTH + consts.SPACING_BETWEEN_WHITE_KEYS) + consts.BLACK_TRIPLET_KEY_OTHER_Y_OFFSET
    bodies.append(
        {
            "name": f"black_key_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "pos": [consts.BLACK_KEY_X_OFFSET, y_coord, consts.BLACK_KEY_Z_OFFSET],
        }
    )
    geoms.append(
        {
            "name": f"black_key_geom_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )
    joints.append(
        {
            "name": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )
    sites.append(
        {
            "name": f"black_key_site_{BLACK_TRIPLET_KEY_INDICES[0]}",
            "dclass": "black_key",
        }
    )

    # Place the twin black keys.
    n = 0
    TWIN_INDICES = list(range(2, consts.NUM_WHITE_KEYS - 1, 7))
    for twin_index in TWIN_INDICES:
        for j in range(2):
            y_coord = -consts.PIANO_LENGTH * 0.5 + (j + 1) * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS) + twin_index * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
            if j == 0:
                y_coord -= consts.BLACK_TWIN_KEY_OTHER_Y_OFFSET
            elif j == 1:
                y_coord += consts.BLACK_TWIN_KEY_OTHER_Y_OFFSET
            bodies.append(
                {
                    "name": f"black_key_{BLACK_TWIN_KEY_INDICES[n]}",
                    "pos": [
                        consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{BLACK_TWIN_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )

            n += 1

    # Place the triplet black keys.
    n = 1  # Skip the lone black key.
    TRIPLET_INDICES = list(range(5, consts.NUM_WHITE_KEYS - 1, 7))
    for triplet_index in TRIPLET_INDICES:
        for j in range(3):
            y_coord = -consts.PIANO_LENGTH * 0.5 + (j + 1) * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS) + triplet_index * (consts.WHITE_KEY_WIDTH + consts.SPACING_BETWEEN_WHITE_KEYS)
            if j == 0:
                y_coord -= consts.BLACK_TRIPLET_KEY_OTHER_Y_OFFSET
            elif j == 2:
                y_coord += consts.BLACK_TRIPLET_KEY_OTHER_Y_OFFSET
            bodies.append(
                {
                    "name": f"black_key_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "pos": [
                        consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{BLACK_TRIPLET_KEY_INDICES[n]}",
                    "dclass": "black_key",
                }
            )

            n += 1

    # Sort the elements based on the key number.
    names: list[str] = [body["name"] for body in bodies]  # type: ignore
    indices = sorted(range(len(names)), key=lambda k: int(names[k].split("_")[-1]))
    bodies = [bodies[i] for i in indices]
    geoms = [geoms[i] for i in indices]
    joints = [joints[i] for i in indices]
    sites = [sites[i] for i in indices]

    # Now create the corresponding MJCF elements and add them to the root.
    for i in range(len(bodies)):
        body = piano_body.add("body", **bodies[i])
        body.add("geom", **geoms[i])
        body.add("joint", **joints[i])
        body.add("site", **sites[i])

    return root


if __name__ == "__main__":
    print(build().to_xml_string())
