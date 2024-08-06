# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import sys
from pathlib import Path

import depthai as dai
from oakutils.blobs import get_model_path
from oakutils.calibration import get_oakd_calibration
from oakutils.nodes import create_stereo_depth, create_xout
from oakutils.nodes.models import create_point_cloud, get_point_cloud_buffer

try:
    from ...device import get_device_count
except ImportError:
    devicefile = Path(__file__).parent.parent.parent / "device.py"
    sys.path.append(str(devicefile.parent))
    from device import get_device_count

try:
    from basic import check_model_equivalence
    from hashs import get_run_tables, write_model_tables, hash_file
except ModuleNotFoundError:
    from .basic import check_model_equivalence
    from .hashs import get_run_tables, write_model_tables, hash_file


def test_create_and_run() -> None:
    if get_device_count() == 0:
        return
    calib_data = get_oakd_calibration()
    hash_table, run_table = get_run_tables()
    for shave in [1, 2, 3, 4, 5, 6]:
        modelname = "pointcloud"
        modelpath = get_model_path(modelname, [], shave)
        model_hash = hash_file(modelpath)
        modelkey = modelpath.stem
        # if the hash is the same and we have already gotten a successful run, continue
        if hash_table[modelkey] == model_hash and run_table[modelkey]:
            continue
        # if the hash is not the same update the hash and set the run to false
        existing_hash = hash_table[modelkey]
        if existing_hash != model_hash:
            hash_table[modelkey] = model_hash
            run_table[modelkey] = False

        pipeline = dai.Pipeline()
        stereo, left, right = create_stereo_depth(pipeline)
        pcl, xin_pcl, device_call = create_point_cloud(
            pipeline, stereo.depth, calib_data, shaves=shave
        )
        xout_pcl = create_xout(pipeline, pcl.out, "pcl_out")

        all_nodes = [
            stereo,
            left,
            right,
            pcl,
            xin_pcl,
            xout_pcl,
        ]
        assert len(all_nodes) == 6
        for node in all_nodes:
            assert node is not None

        with dai.Device(pipeline) as device:
            device_call(device)
            queue: dai.DataOutputQueue = device.getOutputQueue("pcl_out")

            while True:
                data = queue.get()
                pcl_buffer = get_point_cloud_buffer(data)
                assert pcl_buffer is not None
                break

        run_table[modelkey] = True
        write_model_tables(hash_table, run_table)


def test_equivalence() -> None:
    check_model_equivalence("pointcloud")


if __name__ == "__main__":
    test_create_and_run()
    test_equivalence()
