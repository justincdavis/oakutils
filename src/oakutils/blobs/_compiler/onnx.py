import onnx
import onnxsim


def _simplify_onnx(model_path: str, output_path: str, check_num: int = 5):
    """
    Simplifies a model using the onnxsim packages

    Parameters
    ----------
    model_path : str
        The path to the model to simplify
    output_path : str
        The path to save the simplified model to
    check_num : int, optional
        The number of checks to perform on the simplified model, by default 5

    Raises
    ------
    AssertionError
        If the simplified model could not be validated
    """
    model = onnx.load(model_path)
    model_simp, check = onnxsim.simplify(
        model, check_n=check_num, perform_optimization=True
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)


def simplify(model_path: str, output_path: str, check_num: int = 5):
    """
    Simplifies a model using the onnxsim packages

    Parameters
    ----------
    model_path : str
        The path to the model to simplify
    output_path : str
        The path to save the simplified model to
    check_num : int, optional
        The number of checks to perform on the simplified model, by default 5

    Raises
    ------
    AssertionError
        If the simplified model could not be validated
    """
    _simplify_onnx(model_path, output_path, check_num)
