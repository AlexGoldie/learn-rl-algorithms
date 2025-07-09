import ast
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from sympy import Max


def _unparse(node):
    """Simple unparser for AST nodes for Python versions < 3.9"""
    if isinstance(node, ast.Num):
        return str(node.n)
    elif isinstance(node, ast.Name):
        return node.id
    return str(node)


def transform_ast(node, math_functions):
    """Transform AST nodes into JAX operations."""
    if isinstance(node, ast.Call):
        func_name = node.func.id

        if func_name in math_functions:
            # Transform arguments recursively
            args = [transform_ast(arg, math_functions) for arg in node.args]
            # Create a string representing the function call
            args_str = ", ".join(args)
            if func_name == "minus1":
                return f"({args_str} - 1)"

            return f"{math_functions[func_name]}({args_str})"
    elif isinstance(node, ast.BinOp):
        # Handle binary operations
        left = transform_ast(node.left, math_functions)
        right = transform_ast(node.right, math_functions)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[
            type(node.op)
        ]
        return f"({left} {op} {right})"
    elif isinstance(node, ast.Num):
        return str(node.n)
    elif isinstance(node, ast.Name):
        if node.id.startswith("x"):
            try:
                # Extract the index number after 'x'
                index = int(node.id[1:])
                return f"x[...,{index}]"
            except ValueError:
                return node.id
        return node.id
    elif isinstance(node, ast.UnaryOp):
        operand = transform_ast(node.operand, math_functions)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
    return _unparse(node)


def create_jax_function(expression_str):
    """Convert a mathematical expression string into a JAX function."""
    # Define supported mathematical functions
    math_functions = {
        "sin": "jnp.sin",
        "cos": "jnp.cos",
        "exp": "jnp.exp",
        "tanh": "jnp.tanh",
        "square": "jnp.square",
        "neg": "-",
        "abs": "jnp.abs",
        "relu": "nn.relu",
        "min": "jnp.minimum",
        "max": "jnp.maximum",
    }

    # Parse the expression
    tree = ast.parse(expression_str, mode="eval")

    # Transform the AST
    transformed_expr = transform_ast(tree.body, math_functions)

    # Create the function definition with array input
    func_def = f"""
def generated_function(x):    
    import jax.numpy as jnp
    import flax.linen as nn
    return {transformed_expr}
"""

    # Execute the function definition
    namespace = {}
    exec(func_def, namespace)
    generated_function = namespace["generated_function"]
    # Return a partial function with math_functions already bound
    return generated_function
