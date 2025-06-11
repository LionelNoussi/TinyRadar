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
    
    def is_f32_op_rule(self, dir_path: str, file_name: str) -> bool:
        if 'Include' in dir_path:
            return True
        stem = os.path.splitext(file_name)[0]
        forbidden_endings = ('_f16', '_q15', '_q31', '_f64')
        return not any(stem.endswith(end) for end in forbidden_endings)
    
    def not_unnecessary_include_file_rule(self, dir_path: str, file_name: str) -> bool:
        letter = file_name[0]
        if letter.capitalize() == letter:
            return False
        return True


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
    source_root = "/Users/lionel/GitHubRepos/TFLM/CMSIS-DSP"
    project_root = "/Users/lionel/STM32CubeIDE/PrivateWorkspace/SPIExperimentation/"

    # 1. Make proper directories
    destination_root = os.path.join(project_root, 'CMSIS_DSP')
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
        "PrivateInclude/": Destination("PrivateInclude", recursive=True),
        "Include/": Destination("Include", recursive=True),
        "Source/": Destination("Source", recursive=True)
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