#!/usr/bin/python
# graphdef.py
# Creates graphs that can be loaded into a TensorFlow session.

import argparse
import tensorflow as tf

def transform_custom_arg(arg):
  if arg.startswith('--'):
    return arg.strip('-')

  try:
    return int(arg)
  except ValueError:
    pass

  try:
    return float(arg)
  except ValueError:
    pass

  return arg


def cli():
  argparser = argparse.ArgumentParser(prog='graphdef',
                                      description='Creates a graph def')
  argparser.add_argument('--script', required=True,
                         help='The script that defines the graph')
  argparser.add_argument('--output', required=True,
                         help='The output directory to create')
  argparser.add_argument('--text', default=False, action='store_true',
                         help='Whether to output the graph for debugging')

  args, other_args = argparser.parse_known_args()

  other_args_iter = iter(map(lambda arg: transform_custom_arg(arg), other_args))
  other_args = dict(zip(other_args_iter, other_args_iter))

  options = argparse.Namespace()
  options.__dict__.update(other_args)

  return args, options


def main():
  args, options = cli()

  script_scope = {}
  execfile(args.script, script_scope, script_scope)

  graph = script_scope.get('graph', None)
  if graph is not None:
    if not args.text:
      graph_def = graph.as_graph_def().SerializeToString()
    else:
      graph_def = str(graph.as_graph_def())

    file = open(args.output, 'w')
    file.write(graph_def)
    file.close()


main()

