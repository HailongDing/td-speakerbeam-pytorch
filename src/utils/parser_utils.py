# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import argparse
from collections import defaultdict


def prepare_parser_from_dict(dic, parser=None):
    """Prepare argument parser from dictionary.
    
    Args:
        dic (dict): Dictionary with configuration.
        parser (argparse.ArgumentParser, optional): Existing parser.
        
    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    
    def add_arguments_from_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                # Create argument group for nested dictionaries
                group = parser.add_argument_group(key)
                add_arguments_from_dict(value, prefix=f"{key}.")
            else:
                arg_name = f"--{prefix}{key}" if prefix else f"--{key}"
                if isinstance(value, bool):
                    parser.add_argument(arg_name, type=str, default=str(value).lower(),
                                      choices=['true', 'false', 'yes', 'no'])
                elif isinstance(value, (int, float)):
                    parser.add_argument(arg_name, type=type(value), default=value)
                else:
                    parser.add_argument(arg_name, type=str, default=str(value))
    
    add_arguments_from_dict(dic)
    return parser


def parse_args_as_dict(parser, return_plain_args=False):
    """Parse arguments and return as hierarchical dictionary.
    
    Args:
        parser (argparse.ArgumentParser): Argument parser.
        return_plain_args (bool): Whether to also return plain args.
        
    Returns:
        dict or tuple: Hierarchical dictionary (and plain args if requested).
    """
    args = parser.parse_args()
    
    # Convert to hierarchical dictionary
    hierarchical_dict = defaultdict(dict)
    
    for key, value in vars(args).items():
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current_dict = hierarchical_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Convert string booleans back to bool
            if isinstance(value, str) and value.lower() in ['true', 'false', 'yes', 'no']:
                value = value.lower() in ['true', 'yes']
            
            current_dict[parts[-1]] = value
        else:
            # Handle top-level keys
            if isinstance(value, str) and value.lower() in ['true', 'false', 'yes', 'no']:
                value = value.lower() in ['true', 'yes']
            hierarchical_dict['main_args'][key] = value
    
    # Convert defaultdict to regular dict
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        for key, value in d.items():
            if isinstance(value, defaultdict):
                d[key] = convert_defaultdict(value)
        return d
    
    result = convert_defaultdict(hierarchical_dict)
    
    if return_plain_args:
        return result, args
    return result