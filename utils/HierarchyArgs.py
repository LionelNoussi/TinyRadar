import yaml
from typing import get_type_hints, get_origin, get_args
from typing import (
    Union
)


class HierarchyArgs:

    def __init__(self, **kwargs):
        annotations = get_type_hints(self.__class__)
        for key, field_type in annotations.items():
            if key in kwargs:
                val = kwargs[key]

                # Handle "eval:" prefix
                if isinstance(val, str) and val.startswith("eval:"):
                    val = eval(val[len("eval:"):])

                # Convert nested dicts to HierarchyArgs subclasses
                if isinstance(val, dict) and issubclass(field_type, HierarchyArgs):
                    val = field_type(**val)

                # Simple Type check
                self._assert_type(key, val, field_type)

                setattr(self, key, val)
            else:
                setattr(self, key, getattr(self.__class__, key, None))

    def _assert_type(self, key, value, expected_type):
        if value is None:
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            if origin is Union and type(None) in args:
                return
            else:
                raise ValueError(f"Got wrong field type for {key}. Need type {expected_type}")
        
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        if origin is not None and origin is not Union:
            if not isinstance(value, origin):
                raise TypeError(f"Field '{key}' expected {origin}, got {type(value)}")
            if origin in [list, tuple] and args:
                for item in value:
                    if not isinstance(item, args[0]):
                        raise TypeError(f"Field '{key}' items must be {args[0]}, got {type(item)}")
        elif isinstance(value, expected_type) or (expected_type == type(None)):
            return
        elif isinstance(value, HierarchyArgs) and isinstance(expected_type, type) and issubclass(expected_type, HierarchyArgs):
            return
        else:
            raise TypeError(f"Field '{key}' expected {expected_type}, got {type(value)}")

    def to_dict(self):
        result = {}
        for key in get_type_hints(self.__class__):
            val = getattr(self, key)
            if isinstance(val, HierarchyArgs):
                result[key] = val.to_dict()
            else:
                result[key] = val
        return result

    def to_yaml(self, filepath: str):
        def dump_with_comments(obj, indent=0):
            lines = []
            annotations = get_type_hints(obj.__class__)
            for key, hint in annotations.items():
                val = getattr(obj, key)
                # Prepare the YAML representation of the field
                dumped_val = yaml.dump({key: val}, default_flow_style=False).strip()

                # Add type comment at the end of the line
                if isinstance(val, HierarchyArgs):
                    lines.append(" " * indent + f"{key}:")
                    lines.extend(dump_with_comments(val, indent + 2))
                else:
                    try:
                        type_comment = f"  # {hint.__name__}"
                    except:
                        type_comment = f"  # {hint}"
                    for line in dumped_val.splitlines():
                        lines.append(" " * indent + line + type_comment)
            return lines

        lines = dump_with_comments(self)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    @classmethod
    def from_yaml(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def __repr__(self):
        return str(self)
    
    def __str__(self) -> str:
        def format_nested(d, indent=0):
            lines = []
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append("    " * indent + f"{key}:")
                    lines.append(format_nested(value, indent + 1))
                else:
                    lines.append("    " * indent + f"{key}: {value}")
            return "\n".join(lines)
        return format_nested(self.to_dict(), indent=0)