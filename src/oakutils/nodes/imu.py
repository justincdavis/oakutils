# imu = self._pipeline.create(dai.node.IMU)
#         xout_imu = self._pipeline.create(dai.node.XLinkOut)

#         imu.enableIMUSensor(dai.IMUSensor.GRAVITY, self._imu_accelerometer_refresh_rate)
#         imu.enableIMUSensor(
#             dai.IMUSensor.GYROSCOPE_CALIBRATED, self._imu_gyroscope_refresh_rate
#         )
#         imu.setBatchReportThreshold(self._imu_batch_report_threshold)
#         imu.setMaxBatchReports(self._imu_max_batch_reports)

#         xout_imu.setStreamName("imu")

#         imu.out.link(xout_imu.input)

#         self._nodes.extend(["imu"])