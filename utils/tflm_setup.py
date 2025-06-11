import os
import shutil
import inspect
from dataclasses import dataclass

from CommandArgs import CommandLineArgs
from general import cprint, colorize

class CmdArgs(CommandLineArgs):
    reset: bool = False
    no_cmsis: bool = False

@dataclass
class Destination:
    dst_dir: str
    recursive: bool = False
    sub_dirs: tuple = tuple()


class RuleChecker:
    _instance = None
    cmsis_nn_enabled = True
    cmsis_nn_ops = (
        'conv',
        'depthwise_conv',
        'softmax',
        'unidirectional_sequence_lstm',
        'transpose',
        'transpose_conv',
        'fully_connected',
        'pad',
        'mul',
        'README',
        'pooling',
        'svdf',
        'maximum_minimum',
        'batch_matmul',
        'add',
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_valid(cls, dir_path: str, file_name: str) -> bool:
        instance = cls()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if name.endswith('_rule'):
                if not method(dir_path, file_name):
                    return False
        return True

    def is_c_file_rule(self, dir_path: str, file_name: str) -> bool:
        return file_name.endswith(('.c', '.cpp', '.h', '.cc'))

    def file_doesnt_already_exist_rule(self, dir_path: str, file_name: str) -> bool:
        return not os.path.exists(os.path.join(dir_path, file_name))

    def doesnt_end_in_test_rule(self, dir_path: str, file_name: str) -> bool:
        stem = os.path.splitext(file_name)[0]
        return not stem.endswith('test')
    
    def doesnt_include_test_rule(self, dir_path: str, file_name: str) -> bool:
        stem = os.path.splitext(file_name)[0]
        return not '_test_' in stem
    
    def is_not_forbidden_file_rule(self, dir_path: str, file_name: str) -> bool:
        forbidden_files = (
            'irfft_float.cc',
            'irfft_int16.cc',
            'irfft_int32.cc',
            'rfft_float.cc',
            'rfft_int16.cc',
            'rfft_int32.cc',
        )
        return file_name not in forbidden_files
    
    def is_allowed_cmsis_nn_kernel_rule(self, dir_path: str, file_name: str) -> bool:
        if "cmsis_nn" not in dir_path:
            return True
        if not self.cmsis_nn_enabled:
            return False
        return any(file_name.startswith(op) for op in self.cmsis_nn_ops)
    
    def is_overridden_by_cmsis_nn_rule(self, dir_path: str, file_name: str) -> bool:
        if not self.cmsis_nn_enabled:
            return True  # nothing is overridden
        if "micro/kernels" not in dir_path or "cmsis_nn" in dir_path:
            return True  # only restrict standard kernels
        if file_name.endswith('.h'):
            return True
        stem = os.path.splitext(file_name)[0]
        return stem not in self.cmsis_nn_ops  # skip standard kernels that are replaced


def copy_files(src_dir: str, dst_dir: str, recursive: bool):
    print(f"Scanning directory {src_dir}")

    if not os.path.isdir(src_dir):
        cprint(f"  Skipping: {src_dir} does not exist.", 'red')
        return

    os.makedirs(dst_dir, exist_ok=True)

    for entry in os.scandir(src_dir):
        src_path = os.path.join(src_dir, entry.name)
        dst_path = os.path.join(dst_dir, entry.name)

        if entry.is_file():
            if RuleChecker.is_valid(dst_dir, entry.name):
                cprint(f"  All rules passed, copying file {colorize(entry.name, 'white')}", 'green')
                shutil.copy2(src_path, dst_path)
            else:
                cprint(f"  A rule was violated, not copying {colorize(entry.name, 'white')}", 'yellow')
        elif entry.is_dir() and recursive:
            copy_files(src_path, dst_path, recursive)


def main(cmd_args: CmdArgs):
    RESET = cmd_args.reset
    RuleChecker.cmsis_nn_enabled = not cmd_args.no_cmsis

    # This script assumes the following directory structure for your MCU project:

    # Project/
    #     Includes/
    #     Core/
    #     Drivers/
    #     tensorflow_lite/
    #         tensorflow/
    #         third_party/

    # Inside the tensorflow directory will go the code from the tflite-micro GitHub repo and
    # inside the thrid_party directory will go any required third party tools.

    # Source root is the path to the cloned tflite-micro repo
    # destination root is the path to your MCU project
    source_root = "/Users/lionel/GitHubRepos/TFLM/"
    project_root = "/Users/lionel/STM32CubeIDE/PrivateWorkspace/FinalProjctTFLM/"

    # 1. Make proper directories
    destination_root = os.path.join(project_root, 'tensorflow_lite')
    os.makedirs(destination_root, exist_ok=True)

    if RESET:
        print(f"Resetting {destination_root}...")
        for root, dirs, files in os.walk(destination_root, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

    # All paths left are relative to source_root
    # All paths right are relative to destination_root
    paths_to_copy = {
        "tflite-micro/tensorflow/compiler/": Destination('tensorflow/compiler/', recursive=True),
        "tflite-micro/tensorflow/lite/": Destination("tensorflow/lite/"),
        "tflite-micro/tensorflow/lite/c": Destination("tensorflow/lite/c"),
        "tflite-micro/tensorflow/lite/core": Destination("tensorflow/lite/core", recursive=True),
        "tflite-micro/tensorflow/lite/kernels": Destination("tensorflow/lite/kernels", recursive=True),
        "tflite-micro/tensorflow/lite/micro": Destination(
            "tensorflow/lite/micro",
            sub_dirs=('arena_allocator', 'kernels', 'kernels/cmsis_nn/', 'memory_planner', 'models', 'tflite_bridge')),
        "tflite-micro/tensorflow/lite/schema": Destination("tensorflow/lite/schema"),
        "tflite-micro/signal/micro/kernels": Destination("tensorflow/signal/micro/kernels"),
        "tflite-micro/signal/src": Destination("tensorflow/signal/src"),
        "third_party/flatbuffers/include/flatbuffers/": Destination("third_party/flatbuffers/include/flatbuffers/", recursive=True),
        "third_party/gemmlowp/fixedpoint/": Destination("third_party/gemmlowp/fixedpoint/"),
        "third_party/gemmlowp/internal/": Destination("third_party/gemmlowp/internal"),
        "third_party/kissfft": Destination("third_party/kissfft"),
        "third_party/ruy/ruy/profiler/": Destination("third_party/ruy/ruy/ruy/profiler/"),
        "CMSIS-NN/Include/": Destination("CMSIS_NN/Include", recursive=True),
        "CMSIS-NN/Source/": Destination("CMSIS_NN/Source", recursive=True)
    }

    for src_rel, destination in paths_to_copy.items():
        src_dir = os.path.join(source_root, src_rel)
        dst_dir = os.path.join(destination_root, destination.dst_dir)

        # Copy main directory
        copy_files(src_dir, dst_dir, destination.recursive)

        # Copy subdirectories explicitly listed in sub_dirs
        for sub_dir in destination.sub_dirs:
            src_sub_dir = os.path.join(src_dir, sub_dir)
            dst_sub_dir = os.path.join(dst_dir, sub_dir)
            # For subdirs, do a recursive copy always (or adapt if you want)
            copy_files(src_sub_dir, dst_sub_dir, recursive=False)



if __name__ == '__main__':
    cmd_args = CmdArgs()
    main(cmd_args)