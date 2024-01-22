"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete

class Versions:
    v2022_1: str
    v2021_4: str
    v2021_3: str
    v2021_2: str
    ...


def get_filename(url):
    ...

def show_progress(curr, max) -> None:
    ...

class ConfigBuilder:
    precision: Incomplete
    def __init__(self, precision: str = ...) -> None:
        ...
    
    def task_type(self, task_type):
        ...
    
    def framework(self, framework):
        ...
    
    def model_optimizer_args(self, args: list):
        ...
    
    def with_file(self, name, path: Incomplete | None = ..., url: Incomplete | None = ..., google_drive: Incomplete | None = ..., size: Incomplete | None = ..., sha256: Incomplete | None = ...):
        ...
    
    def build(self):
        ...
    


s3: Incomplete
bucket: Incomplete
def set_defaults(url: Incomplete | None = ..., version: Incomplete | None = ..., shaves: Incomplete | None = ..., output_dir: Incomplete | None = ..., compile_params: list = ..., optimizer_params: list = ..., data_type: Incomplete | None = ..., silent: Incomplete | None = ..., zoo_type: Incomplete | None = ..., progress_func: Incomplete | None = ...):
    ...

def is_valid_blob(blob_path):
    ...

class __S3ProgressPercentage:
    def __init__(self, o_s3bucket, key_name) -> None:
        ...
    
    def __call__(self, bytes_amount) -> None:
        ...
    


def compile_blob(blob_name, version: Incomplete | None = ..., shaves: Incomplete | None = ..., req_data: Incomplete | None = ..., req_files: Incomplete | None = ..., output_dir: Incomplete | None = ..., url: Incomplete | None = ..., use_cache: bool = ..., compile_params: Incomplete | None = ..., data_type: Incomplete | None = ..., download_ir: bool = ..., zoo_type: Incomplete | None = ..., dry: bool = ...):
    ...

def from_zoo(name, **kwargs):
    ...

def from_caffe(proto, model, data_type: Incomplete | None = ..., optimizer_params: Incomplete | None = ..., proto_size: Incomplete | None = ..., proto_sha256: Incomplete | None = ..., model_size: Incomplete | None = ..., model_sha256: Incomplete | None = ..., **kwargs):
    ...

def from_onnx(model, data_type: Incomplete | None = ..., optimizer_params: Incomplete | None = ..., model_size: Incomplete | None = ..., model_sha256: Incomplete | None = ..., **kwargs):
    ...

def from_tf(frozen_pb, data_type: Incomplete | None = ..., optimizer_params: Incomplete | None = ..., frozen_pb_size: Incomplete | None = ..., frozen_pb_sha256: Incomplete | None = ..., **kwargs):
    ...

def from_openvino(xml, bin, xml_size: Incomplete | None = ..., xml_sha256: Incomplete | None = ..., bin_size: Incomplete | None = ..., bin_sha256: Incomplete | None = ..., **kwargs):
    ...

def from_config(name, path, **kwargs):
    ...

def zoo_list(version: Incomplete | None = ..., url: Incomplete | None = ..., zoo_type: str = ...):
    ...

def __run_cli__():
    ...

