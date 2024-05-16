from .vpu import check_reconfigure, check_single_model, check_multi_network


def test_vpu():
    check_reconfigure.test_vpu()
    check_single_model.test_vpu()
    check_multi_network.test_vpu()
