from .vpu import check_reconfigure, check_single_model


def test_vpu():
    check_reconfigure.test_vpu()
    check_single_model.test_vpu()
