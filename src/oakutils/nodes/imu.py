# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Module for creating IMU nodes.

Functions
---------
create_imu
    Creates a node for the IMU.
"""
from __future__ import annotations

import depthai as dai


def create_imu(
    pipeline: dai.Pipeline,
    accelerometer_rate: int = 400,
    gyroscope_rate: int = 400,
    batch_report_threshold: int = 1,
    max_batch_reports: int = 10,
    *,
    enable_accelerometer_raw: bool | None = None,
    enable_accelerometer: bool | None = None,
    enable_linear_acceleration: bool | None = None,
    enable_gravity: bool | None = None,
    enable_gyroscope_raw: bool | None = None,
    enable_gyroscope_calibrated: bool | None = None,
    enable_gyroscope_uncalibrated: bool | None = None,
    enable_magnetometer_raw: bool | None = None,
    enable_magnetometer_calibrated: bool | None = None,
    enable_magnetometer_uncalibrated: bool | None = None,
    enable_rotation_vector: bool | None = None,
    enable_game_rotation_vector: bool | None = None,
    enable_geomagnetic_rotation_vector: bool | None = None,
    enable_arvr_stabilized_rotation_vector: bool | None = None,
    enable_arvr_stabilized_game_rotation_vector: bool | None = None,
) -> dai.node.IMU:
    """
    Use to create a node for the IMU.

    Note:
    Sensors which use both gyroscope and accelerometer will default to slower rate.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the IMU to
    accelerometer_rate : int, optional
        The rate of the accelerometer, by default 400
        Options are 100, 200, 400
    gyroscope_rate : int, optional
        The rate of the gyroscope, by default 400
        Options are 125, 250, 400
    batch_report_threshold : int, optional
        The batch report threshold, by default 1
    max_batch_reports : int, optional
        The maximum batch reports, by default 10
    enable_accelerometer_raw : bool, optional
        Enable accelerometer raw, by default False
    enable_accelerometer : bool, optional
        Enable accelerometer, by default False
    enable_linear_acceleration : bool, optional
        Enable linear acceleration, by default False
    enable_gravity : bool, optional
        Enable gravity, by default False
    enable_gyroscope_raw : bool, optional
        Enable gyroscope raw, by default False
    enable_gyroscope_calibrated : bool, optional
        Enable gyroscope calibrated, by default False
    enable_gyroscope_uncalibrated : bool, optional
        Enable gyroscope uncalibrated, by default False
    enable_magnetometer_raw : bool, optional
        Enable magnetometer raw, by default False
    enable_magnetometer_calibrated : bool, optional
        Enable magnetometer calibrated, by default False
    enable_magnetometer_uncalibrated : bool, optional
        Enable magnetometer uncalibrated, by default False
    enable_rotation_vector : bool, optional
        Enable rotation vector, by default False
    enable_game_rotation_vector : bool, optional
        Enable game rotation vector, by default False
    enable_geomagnetic_rotation_vector : bool, optional
        Enable geomagnetic rotation vector, by default False
    enable_arvr_stabilized_rotation_vector : bool, optional
        Enable arvr stabilized rotation vector, by default False
    enable_arvr_stabilized_game_rotation_vector : bool, optional
        Enable arvr stabilized game rotation vector, by default False

    Returns
    -------
    dai.node.IMU
        The IMU node

    Raises
    ------
    ValueError
        If accelerometer_rate is not one of the following: 100, 200, 400
    ValueError
        If gyroscope_rate is not one of the following: 125, 250, 400

    References
    ----------
    https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/
    """
    if enable_accelerometer_raw is None:
        enable_accelerometer_raw = False
    if enable_accelerometer is None:
        enable_accelerometer = False
    if enable_linear_acceleration is None:
        enable_linear_acceleration = False
    if enable_gravity is None:
        enable_gravity = False
    if enable_gyroscope_raw is None:
        enable_gyroscope_raw = False
    if enable_gyroscope_calibrated is None:
        enable_gyroscope_calibrated = False
    if enable_gyroscope_uncalibrated is None:
        enable_gyroscope_uncalibrated = False
    if enable_magnetometer_raw is None:
        enable_magnetometer_raw = False
    if enable_magnetometer_calibrated is None:
        enable_magnetometer_calibrated = False
    if enable_magnetometer_uncalibrated is None:
        enable_magnetometer_uncalibrated = False
    if enable_rotation_vector is None:
        enable_rotation_vector = False
    if enable_game_rotation_vector is None:
        enable_game_rotation_vector = False
    if enable_geomagnetic_rotation_vector is None:
        enable_geomagnetic_rotation_vector = False
    if enable_arvr_stabilized_rotation_vector is None:
        enable_arvr_stabilized_rotation_vector = False
    if enable_arvr_stabilized_game_rotation_vector is None:
        enable_arvr_stabilized_game_rotation_vector = False

    if accelerometer_rate not in [100, 200, 400]:
        err_msg = "accelerometer_rate must be one of the following: 100, 200, 400"
        raise ValueError(
            err_msg,
        )
    if gyroscope_rate not in [125, 250, 400]:
        err_msg = "gyroscope_rate must be one of the following: 125, 250, 400"
        raise ValueError(err_msg)
    slower_rate = min(accelerometer_rate, gyroscope_rate)

    sensors = []
    if enable_accelerometer_raw:
        sensors.append(dai.IMUSensor.ACCELEROMETER_RAW)
    if enable_accelerometer:
        sensors.append(dai.IMUSensor.ACCELEROMETER)
    if enable_linear_acceleration:
        sensors.append(dai.IMUSensor.LINEAR_ACCELERATION)
    if enable_gravity:
        sensors.append(dai.IMUSensor.GRAVITY)
    if enable_gyroscope_raw:
        sensors.append(dai.IMUSensor.GYROSCOPE_RAW)
    if enable_gyroscope_calibrated:
        sensors.append(dai.IMUSensor.GYROSCOPE_CALIBRATED)
    if enable_gyroscope_uncalibrated:
        sensors.append(dai.IMUSensor.GYROSCOPE_UNCALIBRATED)
    if enable_magnetometer_raw:
        sensors.append(dai.IMUSensor.MAGNETOMETER_RAW)
    if enable_magnetometer_calibrated:
        sensors.append(dai.IMUSensor.MAGNETOMETER_CALIBRATED)
    if enable_magnetometer_uncalibrated:
        sensors.append(dai.IMUSensor.MAGNETOMETER_UNCALIBRATED)
    if enable_rotation_vector:
        sensors.append(dai.IMUSensor.ROTATION_VECTOR)
    if enable_game_rotation_vector:
        sensors.append(dai.IMUSensor.GAME_ROTATION_VECTOR)
    if enable_geomagnetic_rotation_vector:
        sensors.append(dai.IMUSensor.GEOMAGNETIC_ROTATION_VECTOR)
    if enable_arvr_stabilized_rotation_vector:
        sensors.append(dai.IMUSensor.ARVR_STABILIZED_ROTATION_VECTOR)
    if enable_arvr_stabilized_game_rotation_vector:
        sensors.append(dai.IMUSensor.ARVR_STABILIZED_GAME_ROTATION_VECTOR)

    imu = pipeline.create(dai.node.IMU)

    # enable the sensors for each type in the sensors list
    for sensor in sensors:
        if sensor in [
            dai.IMUSensor.GYROSCOPE_RAW,
            dai.IMUSensor.GYROSCOPE_CALIBRATED,
            dai.IMUSensor.GYROSCOPE_UNCALIBRATED,
        ]:
            imu.enableIMUSensor(sensor, gyroscope_rate)
        elif sensor in [
            dai.IMUSensor.ACCELEROMETER_RAW,
            dai.IMUSensor.ACCELEROMETER,
            dai.IMUSensor.LINEAR_ACCELERATION,
            dai.IMUSensor.GRAVITY,
        ]:
            imu.enableIMUSensor(sensor, accelerometer_rate)
        else:
            imu.enableIMUSensor(sensor, slower_rate)

    imu.setBatchReportThreshold(batch_report_threshold)
    imu.setMaxBatchReports(max_batch_reports)

    return imu
