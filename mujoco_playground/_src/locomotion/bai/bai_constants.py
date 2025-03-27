"""Defines bipedal bai constants."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "bai"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene.xml"
)

def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

FEET_GEOMS = [
    "left_foot",
    "right_foot",
]

FEET_POS_SENSOR = [f"{site}_pos_in_imu" for site in FEET_SITES]

ROOT_BODY = "base_link"

UPVECTOR_SENSOR = "upvector_imu_torso"
GLOBAL_LINVEL_SENSOR = "global_linvel_imu_torso"
GLOBAL_ANGVEL_SENSOR = "global_angvel_imu_torso"
# imu readout
LOCAL_LINVEL_SENSOR = "local_linvel_imu_torso"
ACCELEROMETER_SENSOR = "accelerometer_imu_torso"
GYRO_SENSOR = "gyro_imu_torso"
