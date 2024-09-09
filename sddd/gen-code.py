dc = {
     "preprocessing": "def process_text(text):\n    # Your code here\n    return text",
     "translations": [],
     "clusters": [
          {
               "name": "JENTACULAR",
               "min_length": 200,
               "max_length": 100,
               "parts_weight": 1.0,
               "regex_weight": 1.0,
               "code": "\ndef parse(line):\n    return  {}\n",
               "parts": [
                    {
                         "max": 1,
                         "min": 1,
                         "name": "BANK"
                    }
               ],
               "patterns": [],
               "children": [
                    {
                         "name": "JENTACULAR2",
                         "min_length": 200,
                         "max_length": 100,
                         "parts_weight": 1.0,
                         "regex_weight": 1.0,
                         "code": "\ndef parse(line):\n    return  {}\n",
                         "parts": [
                              {
                                   "max": 1,
                                   "min": 1,
                                   "name": "BANK"
                              }
                         ],
                         "patterns": [],
                         "children": []
                    }
               ]
          }
     ]
}

sss = f"""
class Parser:
    def __init__(self):
        pass
    
    def preprocesing(self, text):
        return text
    
    def translate(self,text):
        for tmp in self.bundle():
            text = text.replace(tmp["original"], tmp["replacement"])
        return text
    
    def bundle(self):
        return []
    
    def configs(self):
        return []
    
    def hierarchy(self):
        pass
"""
import ast
import astor


def extract_function_node(function_code):
    """
    Parses the function code into an AST node and returns it.
    """
    parsed_code = ast.parse(function_code)
    for node in parsed_code.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("Provided code does not contain a valid function definition.")


def rename_function_node(node, new_name):
    """
    Renames the function node to a new name.
    """
    node.name = new_name
    return node


def generate_parser_code(dc):
    # Extract and modify preprocessing function
    preprocessing_code = dc.get("preprocessing", "def process_text(text):\n    return text")
    preprocessing_node = extract_function_node(preprocessing_code)
    preprocessing_node = rename_function_node(preprocessing_node, 'preprocessing')

    # Generate the bundle method for translations
    translations = dc.get("translations", [])
    bundle_method_code = "def bundle(self):\n    return {\n"
    for translation in translations:
        bundle_method_code += f"        '{translation['Original']}': '{translation['Replacement']}',\n"
    bundle_method_code = bundle_method_code.rstrip(",\n") + "\n    }\n"
    bundle_method_node = ast.parse(bundle_method_code).body[0]

    # Generate configs, hierarchy methods, and parser references
    clusters = dc.get("clusters", [])
    configs_method_code = "def configs(self):\n    return {\n"
    hierarchy_method_code = "@property\ndef hierarchy(self):\n    return {\n"
    parse_methods = []
    parser_reference_code = "def get_parser_for_cluster(self, cluster_name):\n    parsers = {\n"

    def process_cluster(cluster, parent=None, root=False):
        nonlocal configs_method_code, hierarchy_method_code, parse_methods, parser_reference_code

        # Adding cluster configs
        cluster_name = cluster["name"]
        min_length = cluster["min_length"]
        max_length = cluster["max_length"]
        parts_weight = cluster["parts_weight"]
        regex_weight = cluster["regex_weight"]
        cluster_code = cluster["code"]
        parts = cluster.get("parts", [])
        patterns = cluster.get("patterns", [])

        # Include parts and patterns in the configs
        parts_dict = ", ".join([f"'{part['name']}': {{'min': {part['min']}, 'max': {part['max']}}}" for part in parts])
        patterns_list = ", ".join([f"r'{pattern}'" for pattern in patterns])

        configs_method_code += f"        '{cluster_name}': {{\n"
        configs_method_code += f"            'required_parts': {{{parts_dict}}},\n"
        configs_method_code += f"            'regex_patterns': [{patterns_list}],\n"
        configs_method_code += f"            'min_length': {min_length},\n"
        configs_method_code += f"            'max_length': {max_length},\n"
        configs_method_code += f"            'weights': {{'parts': {parts_weight}, 'patterns': {regex_weight}}},\n"
        configs_method_code += f"            'parser': self.parse_{cluster_name.lower()},\n"
        configs_method_code += f"        }},\n"

        # Extract and modify parse function for the cluster
        cluster_parse_node = extract_function_node(cluster_code)
        cluster_parse_node = rename_function_node(cluster_parse_node, f'parse_{cluster_name.lower()}')
        parse_methods.append(cluster_parse_node)

        # Generate a converter method for each cluster
        convert_method_code = f"def convert_{cluster_name.lower()}(self, dc):\n    return dc\n"
        convert_method_node = ast.parse(convert_method_code).body[0]
        parse_methods.append(convert_method_node)

        # Add reference to parsing method in parser reference dictionary
        parser_reference_code += f"        '{cluster_name}': self.parse_{cluster_name.lower()},\n"

        # Handling hierarchy
        if root:
            hierarchy_method_code += f"    '': ['{cluster_name}'],\n"
        else:
            if parent:
                hierarchy_method_code += f"    '{parent}': ['{cluster_name}'],\n"
            else:
                hierarchy_method_code += f"    '{cluster_name}': [],\n"

        # Process child clusters recursively
        for child in cluster.get("children", []):
            process_cluster(child, parent=cluster_name)

    # Process all top-level clusters
    for cluster in clusters:
        process_cluster(cluster, root=True)

    configs_method_code = configs_method_code.rstrip(",\n") + "\n    }\n"
    configs_method_node = ast.parse(configs_method_code).body[0]

    hierarchy_method_code += "    }\n"
    hierarchy_method_node = ast.parse(hierarchy_method_code).body[0]

    # Finish the parser reference method
    parser_reference_code += "    }\n    return parsers.get(cluster_name, None)\n"
    parser_reference_node = ast.parse(parser_reference_code).body[0]

    # Add the specs method
    specs_method_code = """
def specs(self):
    return {
        tmp['name']: tmp for tmp in self.configs().values()
    }
    """
    specs_method_node = ast.parse(specs_method_code).body[0]

    # Generate the complete class definition
    class_def = ast.ClassDef(
        name='GeneratedParser',
        bases=['SRMSParser'],
        keywords=[],
        body=[
                 ast.FunctionDef(
                     name='__init__',
                     args=ast.arguments(
                         args=[ast.arg(arg='self', annotation=None)],
                         vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                     ),
                     body=[ast.Pass()],
                     decorator_list=[]
                 ),
                 preprocessing_node,
                 bundle_method_node,
                 configs_method_node,
                 specs_method_node,
                 hierarchy_method_node,
                 parser_reference_node,
             ] + parse_methods,
        decorator_list=[]
    )

    # Convert AST to code
    module = ast.Module(body=[class_def], type_ignores=[])
    return astor.to_source(module)


# Using the function to generate code
parser_code = generate_parser_code(dc)
print(parser_code)
