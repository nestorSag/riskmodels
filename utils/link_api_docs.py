"""
This module resolves valid anchor tags within a pdoc3 Markdown documentation file for all modules, classes and methods.
"""

from __future__ import annotations
import re
import typing as t
import itertools
import json

import click

NODE_TYPES = ["Module", "Class", "Method", "Variable"]

class PackageTree:

  """This class is used to parse a python package's API documentation into an object tree, where methods and variables belong to classes which belong to modules
  """

  def __init__(
    self, 
    depth:int,
    line: t.Optional[int],
    node_type: str, 
    name: str, 
    children: t.List[PackageTree],
    parent: t.Optional[PackageTree]):
    """Instantiates a tree node
    
    Args:
        depth (int): node depth based on markdown heaaders
        line (t.Optional[int]): line number in original Markdown file
        node_type (str): one of NODE_TYPES
        name (str): fully qualified object name
        children (t.List[PackageTree]): list of child nodes
        parent (t.Optional[PackageTree]): parent node if any
    """
    #
    self.depth = depth
    self.node_type = node_type
    self.name = name
    self.line = line
    self.children = children
    self.parent = parent
    #
    if parent is not None:
      parent.children.append(self)
      #
  def get_reference_string(self) -> str:
    """Produce anchor tag for current node
    """
    if self.node_type == "Module":
      return f"#{self.name}"
    elif self.parent is not None:
      return f"{self.parent.get_reference_string()}.{self.name}"
    else:
      return ""
    #
  def get_formatted_header(self) -> str:
    """Produce node's full Markdown header with valid anchor tag
    
    Args:
        extra_levels (int, optional): Add a number of hashtags to the formatted header. This is useful to downsize every header proportionally in the output document.
    
    Returns:
        str: Description
    """
    return f'{"#" * self.depth} {self.node_type} `{self.name}` {{{self.get_reference_string()}}}'
    #
  def get_ancestor(self, k: int) -> PackageTree:
    """Climb up through ancestor tree
    
    Args:
        k (int): number of levels 
    """
    if k == 0 or self.parent is None:
      return self
    else:
      return self.parent.get_ancestor(k-1)



def parse_package_tree(input_path: str) -> t.List[str]:
  """Takes the path of a valid Markdown file representing a pdoc3 documentation file
  
  Args:
      input_path (str): file path
  
  Returns:
      t.List[str]: markdown file lines with resolved anchor tags
  """
  with open(input_path, "r") as f:
    lines = f.readlines()

  object_pattern = f"(#{{1,}}) `?({'|'.join(NODE_TYPES)}) `?([^`]+)` \\{{#id\\}}"

  tree = PackageTree(
    depth = 0,
    node_type = "Package",
    name = "root",
    line = None,
    children = [],
    parent = None)
  current_depth = 0
  current = tree

  for idx, line in enumerate(lines):
    is_object = re.match(object_pattern, line)
    if is_object:
      depth, node_type, name = len(is_object.group(1)), is_object.group(2), is_object.group(3)
      # integer division by 2 is due to how headers are formatted for modules, classes and methods
      parent = current if depth > current_depth else current.get_ancestor(1 + (current_depth-depth)//2)
      current = PackageTree(
          depth = depth,
          node_type = node_type,
          name = name,
          line = idx,
          children = [],
          parent = parent)
      current_depth = depth
      lines[idx] = current.get_formatted_header()

  return lines

def remap_header_levels(lines: t.List[str], header_map: t.Dict):
  """Remap header depths, i.e. the number of '#' in each header
  
  Args:
      header_map (t.Dict): Dictionary from header depth values to new depth levels
      lines (t.List[str]): Markdown file lines
  """
  pattern_map = {f"^#{{{fro}}} ": "#" * to + " " for fro, to in header_map.items()}
  for idx, line in enumerate(lines):
    for header in pattern_map:
      if re.match(header, line):
        lines[idx] = re.sub(header, pattern_map[header], line)

  return lines

@click.command()
# @click.argument("input_path", type=str)
# @click.argument("output_path", type=str)
@click.option('--input-path', required=True, help='Input file path')
@click.option('--output-path', required=True, help='Output file path')
@click.option('--header-depth-json', default=None, type=str, help='JSON string of int to int maps to remap header depths')
def enhance_api_docs(input_path: str, output_path: str, header_depth_json: str):
  """Takes a pdoc3 Markdown documentation file and resolves valid anchor tags for all modules, classes, methods and variables
  
  Args:
      input_path (str): input Markdown file path
      output_path (str): output Markdown file path
  """
  parsed = parse_package_tree(input_path)
  if header_depth_json is not None:
    header_map = json.loads(header_depth_json)
    header_map = {int(key): int(val) for key, val in header_map.items()}
    parsed = remap_header_levels(parsed, header_map)

  with open(output_path, "w") as f:
    f.write("\n".join(parsed))

if __name__ == "__main__":
  enhance_api_docs()