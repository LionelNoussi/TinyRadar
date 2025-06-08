from typing import Any
import argparse
from typing import get_type_hints
from dataclasses import dataclass


class CommandLineArgs:
    config: str = 'DEFAULT'
    _config_help: str = "The path to the configurations to use. DEFAULT usees the default config, not loaded."
    """
    Simplified class to quickly add command line args.

    Bools are handled differently pay attention.
    """
    def __init__(self) -> None:
        cls = self.__class__
        parser = argparse.ArgumentParser()

        # Iterate over the class attributes
        annotations = {}
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            annotations.update(get_type_hints(base))
        for name, value in annotations.items():
            if name.startswith('_'): continue

            # Find the help string based on naming convention (_<name>_help)
            help_attr = f"_{name}_help"
            help_str = getattr(cls, help_attr, "")

            # Get default value, which is stored as the attribute itself
            default_value = getattr(cls, name, None)
            
            # Handle boolean type argument with store_true/store_false
            if value is bool:
                parser.add_argument(
                    f'--{name}',
                    action='store_true' if not default_value else 'store_false',
                    default=default_value,
                    help=help_str
                )
            else:
                parser.add_argument(
                    f'--{name}', 
                    type=value, 
                    default=default_value, 
                    help=help_str
                )

        # Parse the arguments and assign values to the class instance
        args = parser.parse_args()
        for key, value in vars(args).items():
            setattr(self, key, value)

    def __repr__(self):
        # Generate a concise string representation (useful for debugging)
        args = {name: getattr(self, name) for name in self.__class__.__annotations__.keys()}
        return f"{self.__class__.__name__}({args})"
    
    def __str__(self):
        # Provide a user-friendly summary of the current state of arguments
        args = [f"{name}={getattr(self, name)}" for name in self.__class__.__annotations__.keys()]
        return f"{self.__class__.__name__}({', '.join(args)})"
