from typing import Tuple

import depthai as dai


def create_imu(
    pipeline: dai.Pipeline,
    accelerometer_rate: int = 400,
    gyroscope_rate: int = 400,
    batch_report_threshold: int = 1,
    max_batch_reports: int = 10,
    enable_accelerometer_raw: bool = False,
    enable_accelerometer: bool = False,
    enable_linear_acceleration: bool = False,
    enable_gravity: bool = False,
    enable_gyroscope_raw: bool = False,
    enable_gyroscope_calibrated: bool = False,
    enable_gyroscope_uncalibrated: bool = False,
    enable_magnetometer_raw: bool = False,
    enable_magnetometer_calibrated: bool = False,
    enable_magnetometer_uncalibrated: bool = False,
    enable_rotation_vector: bool = False,
    enable_game_rotation_vector: bool = False,
    enable_geomagnetic_rotation_vector: bool = False,
    enable_arvr_stabilized_rotation_vector: bool = False,
    enable_arvr_stabilized_game_rotation_vector: bool = False,
) -> Tuple[dai.node.IMU, dai.node.XLinkOut]:
    """
    Creates a pipeline for the IMU.
    Sensors which use both gyroscope and accelerometer will default to slower rate.
    An in-depth explanation of the IMU can be found here:
    https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/

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
    dai.node.XLinkOut
        The output node, with stream name "imu"

    Raises
    ------
    ValueError
        If accelerometer_rate is not one of the following: 100, 200, 400
    ValueError
        If gyroscope_rate is not one of the following: 125, 250, 400
    """
    if accelerometer_rate not in [100, 200, 400]:
        raise ValueError(
            "accelerometer_rate must be one of the following: 100, 200, 400"
        )
    if gyroscope_rate not in [125, 250, 400]:
        raise ValueError("gyroscope_rate must be one of the following: 125, 250, 400")
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
    xout_imu = pipeline.create(dai.node.XLinkOut)

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

    xout_imu.setStreamName("imu")

    imu.out.link(xout_imu.input)

    return imu, xout_imu
