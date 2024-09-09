import json
import pickle
import random
import re
import traceback

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox, QSpinBox, QLineEdit, QFormLayout, QVBoxLayout, QListWidgetItem, QDialog, \
    QTableWidgetItem, QFileDialog, QMessageBox

from app import Ui_MainWindow
from pyqode.python.backend import server
from pyqode.python.widgets import PyCodeEdit

import ast
import astor
import logging

# Set the logging level for pyqode to CRITICAL to suppress less severe messages
logging.getLogger('pyqode').setLevel(logging.CRITICAL)

# If necessary, apply this to other relevant loggers as well
logging.getLogger('pyqode.core.backend.server').setLevel(logging.CRITICAL)
logging.getLogger('pyqode.python.backend').setLevel(logging.CRITICAL)


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


def add_staticmethod_decorator(node):
    """
    Adds a @staticmethod decorator to the function node.
    """
    decorator = ast.Name(id='staticmethod', ctx=ast.Load())
    node.decorator_list.insert(0, decorator)
    return node


SUPER_CLASS = '''
import re
from datetime import datetime

class SRMSParser:
    def split_into_chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_configs(self):
        return {}

    def get_bundle(self):
        return {}

    def apply_bundle_replacements(self, line):
        for k, v in self.get_bundle().items():
            line = line.replace(k, v)
        return line

    def split_by_indices(self, input_list, indices):
        indices = sorted(indices)
        return [input_list[i:j] for i, j in zip([0] + indices, indices + [None])]

    def traverse_hierarchy(self, dc, hierarchy, current_tag, path=None):
        path = path or []
        indices = [idx for idx, entry in enumerate(dc) if entry["TAG"] == current_tag]
        split_lists = self.split_by_indices(dc, indices)[1:]
        processed = []

        for sublist in split_lists:
            first_element = sublist[0]
            new_path = path + [first_element]
            processed.append(new_path)
            for tag in hierarchy.get(first_element['TAG'], []):
                processed.extend(self.traverse_hierarchy(sublist[1:], hierarchy, tag, new_path))
        return processed

    def is_dict_subset(self, smaller, larger):
        return all(item in larger.items() for item in smaller.items())

    def simplify_dictionary_list(self, dicts):
        simplified = []
        for d in dicts:
            if not any(self.is_dict_subset(d, s) for s in simplified):
                simplified = [s for s in simplified if not self.is_dict_subset(s, d)]
                simplified.append(d)
        return simplified

    def find_previous_tag(self, tag, hierarchy):
        return next((k for k, v in hierarchy.items() if tag in v), None)

    def merge_and_simplify_dicts(self, dc, hierarchy, last_tag):
        prev_tag = self.find_previous_tag(last_tag, hierarchy)
        if not prev_tag:
            return dc

        reversed_dc = list(reversed(dc))
        updated_dc = []

        while prev_tag:
            merged_list = []
            skip = False
            for idx, line in enumerate(reversed_dc):
                if line["TAG"] == last_tag and not skip:
                    for i in range(idx + 1, len(reversed_dc)):
                        if reversed_dc[i]["TAG"] == prev_tag:
                            line.update(reversed_dc[i])
                            break
                    merged_list.append(line)
                    skip = True
                elif line["TAG"] not in {prev_tag, last_tag}:
                    merged_list.append(line)

            reversed_dc = merged_list
            last_tag, prev_tag = prev_tag, self.find_previous_tag(prev_tag, hierarchy)

        return list(reversed(reversed_dc))

    def sanitize_string(self, key):
        return ''.join('_' if not char.isalnum() else char for char in key).strip('_')

    def calculate_string_similarity(self, string, required_parts, regex_patterns, min_length=None, max_length=None,
                                    case_insensitive=False, include_whitespace=False, detailed_report=False,
                                    weights=None, prev=None, prev_contains=[]):
        if prev and prev_contains:
            if all(tmp in prev for tmp in prev_contains):
                return {"similarity_score": 1.0, "reason": "Previous line contains tokens"}
            if all(tmp not in prev for tmp in prev_contains):
                return {"similarity_score": 0.0, "reason": "Previous line does not contain tokens"}

        if case_insensitive:
            string = string.lower()
            required_parts = {k.lower(): v for k, v in required_parts.items()}
            regex_patterns = [pattern.lower() for pattern in regex_patterns]

        effective_string = re.sub(r'\s+', '', string) if not include_whitespace else string
        effective_length = len(effective_string)

        if (min_length and effective_length < min_length) or (max_length and effective_length > max_length):
            if detailed_report:
                return {"similarity_score": 0.0, "reason": "Length constraint not met"}
            return 0.0
        part_score = all(min_occurrences <= len(re.findall(part, string)) <= max_occurrences
                         for part, constraints in required_parts.items()
                         for min_occurrences, max_occurrences in [(constraints.get('min', 1), constraints.get('max', float('inf')))])

        matched_patterns = [pattern for pattern in regex_patterns if re.search(pattern, string)]
        pattern_score = len(matched_patterns) / len(regex_patterns) if regex_patterns else 1.0

        if weights:
            part_weight, pattern_weight = weights.get('parts', 1), weights.get('patterns', 1)
            total_weight = part_weight + pattern_weight
            similarity_score = (part_weight * part_score + pattern_weight * pattern_score) / total_weight
        else:
            similarity_score = pattern_score

        if detailed_report:
            return {
                "similarity_score": similarity_score,
                "matched_patterns": matched_patterns,
                "total_patterns": len(regex_patterns)
            }

        return similarity_score

    def clean_redundant_spaces(self, s, iterations=20):
        for _ in range(iterations):
            s = s.replace("  ", " ")
        return s

    def preprocess_and_apply_bundle(self, string):
        """Preprocess and apply bundle replacements to the input string."""
        string = self.preprocessing(string)
        string = self.apply_bundle_replacements(string).strip()
        return string

    def calculate_similarity_scores(self, string):
        """Calculate similarity scores for the input string against all configurations."""
        scores = {
            tag: self.calculate_string_similarity(
                string,
                config["required_parts"],
                config["regex_patterns"],
                config["min_length"],
                config["max_length"],
                config["case_insensitive"],
                config["include_whitespace"],
                config["detailed_report"],
                config["weights"]
            )['similarity_score']
            for tag, config in self.configs().items()
        }
        return scores

    def parse_and_tag_line(self, string, scores):
        """Parse the line and tag it based on the highest similarity score."""
        best_tag = max(scores, key=scores.get)
        parsed_line = {}

        if scores[best_tag] > 0.8:
            cleaned_string = self.clean_redundant_spaces(string)
            tagged_line = f'{best_tag}#{cleaned_string}'
            parser_function = self.get_parser_for_cluster(best_tag)
            parsed_line = parser_function(tagged_line)
            parsed_line["TAG"] = best_tag

        return parsed_line

    def tag_document(self, document, leaf):
        """Process the entire document and tag each line."""
        parsed = []
        for line in document.split('\\n'):
            processed_line = self.preprocess_and_apply_bundle(line)
            if not processed_line:
                continue
            scores = self.calculate_similarity_scores(processed_line)
            parsed_line = self.parse_and_tag_line(processed_line, scores)

            if parsed_line:
                parsed.append(parsed_line)

        return self.merge_and_simplify_dicts(parsed, self.hierarchy, leaf)

    def get_parsed_lines(self, document):
        """Mirror of tag_document, returns parsed lines without merging or simplifying."""
        parsed = []
        for line in document.split('\\n'):
            processed_line = self.preprocess_and_apply_bundle(line)
            if not processed_line:
                continue

            scores = self.calculate_similarity_scores(processed_line)
            parsed_line = self.parse_and_tag_line(processed_line, scores)

            if parsed_line:
                parsed.append(parsed_line)

        return parsed

    def get_line_scores(self, document):
        """Return a map between each line and its similarity scores."""
        line_scores = {}
        for line in document.split('\\n'):
            processed_line = self.preprocess_and_apply_bundle(line)
            if not processed_line:
                continue

            scores = self.calculate_similarity_scores(processed_line)
            best_tag = max(scores, key=scores.get)
            parsed_line = {}

            if scores[best_tag] < 0.8:
                best_tag = "ABADON"
                
            line_scores[processed_line] = best_tag

        return line_scores

    def merge_elements(self, list_of_dicts, source_index, target_index):
        try:
            # Ensure indices are within bounds of the list
            if source_index < 0 or source_index >= len(list_of_dicts) or target_index < 0 or target_index >= len(
                    list_of_dicts):
                raise IndexError("Index out of range.")

            # Merge the dictionaries: update dict at target_index with contents of dict at source_index
            tag_value = list_of_dicts[target_index]["TAG"]
            list_of_dicts[target_index].update(list_of_dicts[source_index])
            list_of_dicts[target_index]["TAG"] = tag_value

            return list_of_dicts

        except IndexError as e:
            print(f"Error: {e}")
            return list_of_dicts

    def set_parent_for_child(self, node, parent_name=None):
        if len(node["children"]) == 0:
            node["_parent"] = parent_name
            return node
        else:
            for child_node in node["children"]:
                return self.set_parent_for_child(child_node, parent_name=node["name"])

    def assign_parents_to_children(self, node_list):
        nodes_with_parents = []
        for node in node_list:
            nodes_with_parents.append(self.set_parent_for_child(node))
        return nodes_with_parents

    def remove_tag_from_structure(self, structure, tag_to_remove):
        updated_structure = []

        for element in structure:
            if element.get('name') != tag_to_remove:
                if 'children' in element and isinstance(element['children'], list):
                    element['children'] = self.remove_tag_from_structure(element['children'], tag_to_remove)
                updated_structure.append(element)

        return updated_structure

    def process_elements(self, structure_data, hierarchy_list):
        reversed_data_lines = structure_data.split('\\n')[::-1]
        structure_dicts = [eval(line) for line in reversed_data_lines]

        previous_structure = []
        while len(hierarchy_list) > 0:
            child_elements = self.assign_parents_to_children(hierarchy_list)
            for child in child_elements:
                hierarchy_list = self.remove_tag_from_structure(hierarchy_list, child["name"])
                merge_positions = []
                for idx, element_dict in enumerate(structure_dicts):
                    if element_dict["TAG"] == child["name"]:
                        for idx2, potential_parent_dict in enumerate(structure_dicts):
                            if idx2 > idx and potential_parent_dict["TAG"] == child["_parent"]:
                                merge_positions.append((idx, idx2))
                                break

                for idx, merge_pair in enumerate(merge_positions):
                    structure_dicts = self.merge_elements(structure_dicts, merge_pair[0], merge_pair[1])

                previous_structure = structure_dicts
                structure_dicts = [entry for entry in structure_dicts if entry['TAG'] != child["name"]]

        return previous_structure

    def process_chunk(self, chunk, hierarchy_branch):

        for child_branch in hierarchy_branch["children"]:
            merge_positions = []
            for idx, element in enumerate(chunk):
                if element['TAG'] == hierarchy_branch["name"]:
                    for idx2, child_element in enumerate(chunk):
                        if child_element['TAG'] == child_branch["name"]:
                            merge_positions.append((idx, idx2))
            for parent_idx, child_idx in merge_positions:
                tag_value = chunk[child_idx]["TAG"]
                chunk[child_idx].update(chunk[parent_idx])
                chunk[child_idx]["TAG"] = tag_value
                self.process_chunk(chunk[1:], child_branch)
        return chunk

    def find_sector_indices(self, data_lines, start_tag):
        sector_indices = []
        for idx, line in enumerate(data_lines):
            if line['TAG'] == start_tag:
                sector_indices.append(idx)
        return sector_indices

    def divide_into_chunks(self, data_lines, sector_indices):
        chunks = []

        for i in range(len(sector_indices) - 1):
            start_idx = sector_indices[i]
            end_idx = sector_indices[i + 1]
            chunks.append(data_lines[start_idx:end_idx])
        return chunks

    def gather_all_child_elements(self, node, child_list):
        for child in node["children"]:
            if len(child["children"]) == 0:
                child_list.append(child)
            else:
                self.gather_all_child_elements(child, child_list)

    def process_elements_with_hierarchy(self, structure_data, hierarchy, start_tag):
        data_lines = structure_data.split('\\n')
        data_lines = [eval(line) for line in data_lines]
        sector_indices = self.find_sector_indices(data_lines, start_tag)
        chunks = self.divide_into_chunks(data_lines, sector_indices)
        all_children = []
        self.gather_all_child_elements(hierarchy[0], all_children)
        child_tags = [child["name"] for child in all_children]
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self.process_chunk(chunk, hierarchy[0])
            chunk_filtered = [entry for entry in chunk if entry["TAG"] in child_tags]
            processed_chunks.append(chunk_filtered)
        return processed_chunks
'''


def generate_parser_code(dc):
    # Extract and modify preprocessing function
    preprocessing_code = dc.get("preprocessing", "def process_text(text):\n    return text")
    preprocessing_node = extract_function_node(preprocessing_code)
    preprocessing_node = rename_function_node(preprocessing_node, 'preprocessing')
    preprocessing_node = add_staticmethod_decorator(preprocessing_node)

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
        configs_method_code += f"            'case_insensitive': {False},\n"
        configs_method_code += f"            'include_whitespace': {False},\n"
        configs_method_code += f"            'detailed_report': {True},\n"
        configs_method_code += f"            'weights': {{'parts': {parts_weight}, 'patterns': {regex_weight}}},\n"
        configs_method_code += f"            'parser': self.parse_{cluster_name.lower()},\n"
        configs_method_code += f"        }},\n"

        # Extract and modify parse function for the cluster
        cluster_parse_node = extract_function_node(cluster_code)
        cluster_parse_node = rename_function_node(cluster_parse_node, f'parse_{cluster_name.lower()}')
        cluster_parse_node = add_staticmethod_decorator(cluster_parse_node)
        parse_methods.append(cluster_parse_node)

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

    MODULE = r"""
if __name__ == "__main__":
    parser = GeneratedParser()
    
    document_text = ""
    parsed_lines = parser.get_parsed_lines(document_text)

    parsed_text = "\n".join([str(line) for line in parsed_lines])
    tech = "Up2Down"
    if tech == "Up2Down":
        results = parser.process_elements_with_hierarchy(parsed_text, [parser.configs()], "JENTACULAR")
        results = [s for t in results for s in t]
    else:
        results = parser.process_elements(parsed_text, [parser.configs()])
"""
    return SUPER_CLASS + astor.to_source(module)


class TranslationDialog(QtWidgets.QDialog):
    def __init__(self, original_text="", replacement_text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add/Edit Translation")
        self.setModal(True)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.formLayout = QtWidgets.QFormLayout()
        self.layout.addLayout(self.formLayout)

        self.originalLineEdit = QtWidgets.QLineEdit(self)
        self.replacementLineEdit = QtWidgets.QLineEdit(self)

        self.formLayout.addRow("Original:", self.originalLineEdit)
        self.formLayout.addRow("Replacement:", self.replacementLineEdit)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                                                    self)
        self.layout.addWidget(self.buttonBox)

        self.originalLineEdit.setText(original_text)
        self.replacementLineEdit.setText(replacement_text)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def get_data(self):
        return self.originalLineEdit.text(), self.replacementLineEdit.text()


class PartDialog(QtWidgets.QDialog):
    """
    Dialog to input part name, min value, and max value together.
    """

    def __init__(self, part_name="", min_val=1, max_val=1, parent=None, det="Part"):
        super().__init__(parent)
        self.setWindowTitle(f"Add/Edit {det}")
        self.layout = QVBoxLayout()

        form_layout = QFormLayout()

        self.part_name_input = QLineEdit(part_name)
        form_layout.addRow(f"{det}:", self.part_name_input)

        self.min_value_input = QSpinBox()
        self.min_value_input.setRange(0, 1000)
        self.min_value_input.setValue(min_val)
        form_layout.addRow("Min Value:", self.min_value_input)

        self.max_value_input = QSpinBox()
        self.max_value_input.setRange(0, 1000)
        self.max_value_input.setValue(max_val)
        form_layout.addRow("Max Value:", self.max_value_input)

        self.layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

    def getData(self):
        part_name = self.part_name_input.text()
        min_val = self.min_value_input.value()
        max_val = self.max_value_input.value()
        return part_name, min_val, max_val


class ParserApp(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        self.code_editor = PyCodeEdit(self.tabPreprocessing)
        self.code_editor.backend.start(server.__file__)
        self.codePlaceholder.addWidget(self.code_editor)

        self.clustering_code_editor = PyCodeEdit(self.tabPreprocessing)
        self.clustering_code_editor.backend.start(server.__file__)
        self.clustering_code_editor.setMinimumSize(QtCore.QSize(0, 200))
        self.clustering_code_editor.setPlainText("""
def parse(line):
    return  {}
""")
        self.clusterCodeWidget.addWidget(self.clustering_code_editor)

        self.clusteringTreeWidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.clusteringTreeWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.clusteringTreeWidget.setDragEnabled(True)
        self.clusteringTreeWidget.setAcceptDrops(True)
        self.clusteringTreeWidget.setDropIndicatorShown(True)

        self.code_editor2 = PyCodeEdit(self.tabExport)
        self.code_editor2.backend.start(server.__file__)
        self.clusteringVerticalLayout2.addWidget(self.code_editor2)

        self.init_preprocessing_combobox()
        self.init_buttons()
        self.init_actions()
        self.clusteringTreeWidget.itemClicked.connect(self.on_tree_item_clicked)
        self.parserComboBox.currentIndexChanged.connect(self.on_parserComboBox_change)

    def on_parserComboBox_change(self):
        selected_data = self.parserComboBox.itemData(self.parserComboBox.currentIndex(), role=Qt.UserRole)
        self.clustering_code_editor.setPlainText(selected_data)

    def init_actions(self):
        self.actionNewProject.triggered.connect(self.on_new_project)
        self.actionSaveProject.triggered.connect(self.on_save_project)
        self.actionOpenProject.triggered.connect(self.on_open_project)

    def on_open_project(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.MainWindow, "Open Project", "",
                                                   "PICKLE Files (*.pkl);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                with open(file_name, 'rb') as file:
                    project = pickle.load(file)

                    # Populate the UI with the loaded data
                    self.lineEditMainText.setText(project.get("main_text_file", ""))
                    self.spinBoxMainTextLimit.setValue(project.get("main_text_length", 0))
                    self.textEditMainText.setPlainText(project.get("main_text_content", ""))

                    self.lineEditAlternativeText.setText(project.get("alternative_text_file", ""))
                    self.spinBoxAlternativeTextLimit.setValue(project.get("alternative_text_length", 0))
                    self.textEditAlternativeText.setPlainText(project.get("alternative_text_content", ""))

                    self.code_editor.setPlainText(project.get("preprocessing_code", ""))
                    self.list_of_dicts_to_qtablewidget(project.get("translations", []), self.translationsTableWidget)
                    self.populate_tree_from_dict(project.get("clusters", []), self.clusteringTreeWidget)

                    self.code_editor2.setPlainText(project.get("parser_code", ""))


            except Exception as e:
                QMessageBox.critical(self.MainWindow, "Load Error",
                                     f"An error occurred while loading the project:\n{str(e)}")

    def on_save_project(self):

        project = {
            "main_text_file": self.lineEditMainText.text(),
            "main_text_content": self.textEditMainText.toPlainText(),
            "main_text_length": self.spinBoxMainTextLimit.value(),

            "alternative_text_file": self.lineEditAlternativeText.text(),
            "alternative_text_content": self.textEditAlternativeText.toPlainText(),
            "alternative_text_length": self.spinBoxAlternativeTextLimit.value(),

            "preprocessing_code": self.code_editor.toPlainText(),
            "translations": self.qtablewidget_to_list_of_dicts(self.translationsTableWidget),
            "clusters": self.tree_to_dict(self.clusteringTreeWidget),
            "parser_code": self.code_editor2.toPlainText()
        }

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self.MainWindow, "Save Project", "",
                                                   "PICKLE Files (*.pkl);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                with open(file_name, 'wb') as file:
                    pickle.dump(project, file)
                return True
            except Exception as e:
                QMessageBox.critical(self.MainWindow, "Save Error",
                                     f"An error occurred while saving the project:\n{str(e)}")
                return False

    def on_new_project(self):
        self.project = {
            "main_text_file": "",
            "main_text_content": "",
            "main_text_length": 0,

            "alternative_text_file": "",
            "alternative_text_content": "",
            "alternative_text_length": 0,

            "preprocessing_code": "",
            "translations": [],
            "clusters": [],
            "parser_code": ""
        }

        self.lineEditMainText.setText(self.project["main_text_file"])
        self.spinBoxMainTextLimit.setValue(self.project["main_text_length"])
        self.textEditMainText.setText(self.project["main_text_content"])

        self.lineEditAlternativeText.setText(self.project["alternative_text_file"])
        self.spinBoxAlternativeTextLimit.setValue(self.project["alternative_text_length"])
        self.textEditAlternativeText.setText(self.project["alternative_text_content"])

        self.code_editor.setPlainText(self.project["preprocessing_code"])
        self.list_of_dicts_to_qtablewidget(self.project["translations"], self.translationsTableWidget)
        self.populate_tree_from_dict(self.project["clusters"], self.clusteringTreeWidget)

        self.code_editor2.setPlainText(self.project["parser_code"])

    def init_buttons(self):
        # Connecting buttons to their respective methods
        self.openMainTextButton.clicked.connect(self.on_open_main_text_clicked)
        self.reloadMainTextButton.clicked.connect(self.on_reload_main_text_clicked)
        self.openAlternativeTextButton.clicked.connect(self.on_open_alternative_text_clicked)
        self.reloadAlternativeTextButton.clicked.connect(self.on_reload_alternative_text_clicked)

        self.clearPreprocessingButton.clicked.connect(self.on_clear_preprocessing_clicked)
        self.applyPreprocessingButton.clicked.connect(self.on_apply_preprocessing_clicked)

        self.clearTranslationsButton.clicked.connect(self.on_clear_translations_clicked)
        self.applyTranslationsButton.clicked.connect(self.on_apply_translations_clicked)
        self.addTranslationButton.clicked.connect(self.on_add_translation_clicked)
        self.editTranslationButton.clicked.connect(self.on_edit_translation_clicked)
        self.deleteTranslationButton.clicked.connect(self.on_delete_translation_clicked)

        self.clearMappingsButton.clicked.connect(self.on_clear_mappings_clicked)
        self.applyMappingsButton.clicked.connect(self.on_apply_mappings_clicked)
        self.addMappingButton.clicked.connect(self.on_add_mapping_clicked)
        self.editMappingButton.clicked.connect(self.on_edit_mapping_clicked)
        self.deleteMappingButton.clicked.connect(self.on_delete_mapping_clicked)


        self.clearClusteringButton.clicked.connect(self.on_clear_clustering_clicked)
        self.applyClusteringButton.clicked.connect(self.on_apply_clustering_clicked)
        self.addClusterButton.clicked.connect(self.on_add_cluster_clicked)
        self.editClusterButton.clicked.connect(self.on_edit_cluster_clicked)
        self.deleteClusterButton.clicked.connect(self.on_delete_cluster_clicked)
        self.exportClusterButton.clicked.connect(self.on_export_cluster_clicked)
        self.generateClusterButton.clicked.connect(self.on_generate_cluster_clicked)
        self.runTaggerButton.clicked.connect(self.on_run_tagger_clicked)
        self.runParserButton.clicked.connect(self.on_run_parser_clicked)
        self.runParserButton2.clicked.connect(self.on_run_parser_and_merge_clicked)

        self.addPartButton.clicked.connect(self.on_add_part_clicked)
        self.editPartButton.clicked.connect(self.on_edit_part_clicked)
        self.deletePartButton.clicked.connect(self.on_delete_part_clicked)
        self.addPatternButton.clicked.connect(self.on_add_pattern_clicked)
        self.editPatternButton.clicked.connect(self.on_edit_pattern_clicked)
        self.deletePatternButton.clicked.connect(self.on_delete_pattern_clicked)

        self.toggle_button2.clicked.connect(self.toggle_view2)
        self.toggle_button1.clicked.connect(self.toggle_view)

    def toggle_view(self):
        if self.textEditMainText.isVisible():
            self.textEditMainText.hide()
            self.web_viewMainText.show()
        else:
            self.web_viewMainText.hide()
            self.textEditMainText.show()

    def toggle_view2(self):
        if self.textEditAlternativeText.isVisible():
            self.textEditAlternativeText.hide()
            self.web_viewAlternativeText.show()
        else:
            self.web_viewAlternativeText.hide()
            self.textEditAlternativeText.show()

    def tree_to_dict(self, tree_widget):
        """
        Convert the data in the QTreeWidget to a list of dictionaries,
        with nested dictionaries representing the tree structure.
        """

        def item_to_dict(item):
            """
            Recursively convert a QTreeWidgetItem and its children to a dictionary.
            """
            cluster_data = item.data(0, QtCore.Qt.UserRole)
            item_dict = {
                "name": cluster_data.get("name"),
                "min_length": cluster_data.get("min_length"),
                "max_length": cluster_data.get("max_length"),
                "parts_weight": cluster_data.get("parts_weight"),
                "regex_weight": cluster_data.get("regex_weight"),
                "code": cluster_data.get("code"),
                "parts": cluster_data.get("parts"),
                "patterns": cluster_data.get("patterns"),
                "children": []
            }

            for i in range(item.childCount()):
                child_item = item.child(i)
                item_dict["children"].append(item_to_dict(child_item))

            return item_dict

        root = tree_widget.invisibleRootItem()
        tree_data = []

        for i in range(root.childCount()):
            top_item = root.child(i)
            tree_data.append(item_to_dict(top_item))

        return tree_data

    def qtablewidget_to_list_of_dicts(self, table_widget):
        headers = [table_widget.horizontalHeaderItem(i).text() for i in range(table_widget.columnCount())]
        data_list = []

        for row in range(table_widget.rowCount()):
            row_data = {}
            for column in range(table_widget.columnCount()):
                item = table_widget.item(row, column)
                row_data[headers[column]] = item.text() if item is not None else None
            data_list.append(row_data)

        return data_list

    def list_of_dicts_to_qtablewidget(self, data_list, table_widget):
        # Check if the data_list is not empty
        if not data_list:
            return

        # Extract headers from the first dictionary keys
        headers = list(data_list[0].keys())

        # Set column count and headers
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)

        # Set row count based on the number of dictionaries in the list
        table_widget.setRowCount(len(data_list))

        # Populate the QTableWidget with data
        for row, row_data in enumerate(data_list):
            for column, header in enumerate(headers):
                item_text = row_data.get(header, "")
                table_widget.setItem(row, column, QTableWidgetItem(item_text))

    def populate_tree_from_dict(self, data_list, parent_item=None):
        """
        Recursively populate a QTreeWidget with data from a list of dictionaries.

        :param data_list: List of dictionaries containing tree data.
        :param parent_item: The parent QTreeWidgetItem to attach the new items to.
                            If None, items are added to the top level of the tree.
        """
        for data in data_list:
            # Create a new QTreeWidgetItem for this dictionary
            if parent_item is None:
                # If no parent_item is specified, add to the top level of the tree
                item = QtWidgets.QTreeWidgetItem(self.clusteringTreeWidget)
            else:
                # If a parent_item is provided, add as a child of the parent
                item = QtWidgets.QTreeWidgetItem(parent_item)

            # Set the item text and data
            item.setText(0, data["name"])
            item.setData(0, QtCore.Qt.UserRole, data)

            # Recursively add children if they exist
            if "children" in data and data["children"]:
                self.populate_tree_from_dict(data["children"], item)

    def on_add_mapping_clicked(self):
        # Get selected text from main and alternative text areas
        selected_main_text = self.textEditMainText.textCursor().selectedText()
        selected_alternative_text = self.textEditAlternativeText.textCursor().selectedText()

        # Open the dialog with the selected text
        dialog = TranslationDialog(selected_main_text, selected_alternative_text, self.tabWidget)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            original, replacement = dialog.get_data()
            self.add_mapping_to_table(original, replacement)

    def on_add_translation_clicked(self):
        # Get selected text from main and alternative text areas
        selected_main_text = self.textEditMainText.textCursor().selectedText()
        selected_alternative_text = self.textEditAlternativeText.textCursor().selectedText()

        # Open the dialog with the selected text
        dialog = TranslationDialog(selected_main_text, selected_alternative_text, self.tabWidget)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            original, replacement = dialog.get_data()
            self.add_translation_to_table(original, replacement)

    def on_edit_mapping_clicked(self):
        # Get the currently selected row in the table
        current_row = self.mappingsTableWidget.currentRow()
        if current_row < 0:
            return

        # Get existing values
        original_item = self.mappingsTableWidget.item(current_row, 0)
        replacement_item = self.mappingsTableWidget.item(current_row, 1)

        original_text = original_item.text() if original_item else ""
        replacement_text = replacement_item.text() if replacement_item else ""

        # Open the dialog with the current values
        dialog = TranslationDialog(original_text, replacement_text, self.tabWidget)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            original, replacement = dialog.get_data()
            self.edit_mapping_in_table(current_row, original, replacement)

    def on_delete_mapping_clicked(self):
        # Get the currently selected row and delete it
        current_row = self.mappingsTableWidget.currentRow()
        if current_row >= 0:
            self.mappingsTableWidget.removeRow(current_row)

    def on_edit_translation_clicked(self):
        # Get the currently selected row in the table
        current_row = self.translationsTableWidget.currentRow()
        if current_row < 0:
            return

        # Get existing values
        original_item = self.translationsTableWidget.item(current_row, 0)
        replacement_item = self.translationsTableWidget.item(current_row, 1)

        original_text = original_item.text() if original_item else ""
        replacement_text = replacement_item.text() if replacement_item else ""

        # Open the dialog with the current values
        dialog = TranslationDialog(original_text, replacement_text, self.tabWidget)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            original, replacement = dialog.get_data()
            self.edit_translation_in_table(current_row, original, replacement)

    def on_delete_translation_clicked(self):
        # Get the currently selected row and delete it
        current_row = self.translationsTableWidget.currentRow()
        if current_row >= 0:
            self.translationsTableWidget.removeRow(current_row)

    def add_mapping_to_table(self, original, replacement):
        row_position = self.mappingsTableWidget.rowCount()
        self.mappingsTableWidget.insertRow(row_position)
        self.mappingsTableWidget.setItem(row_position, 0, QtWidgets.QTableWidgetItem(original))
        self.mappingsTableWidget.setItem(row_position, 1, QtWidgets.QTableWidgetItem(replacement))

    def add_translation_to_table(self, original, replacement):
        row_position = self.translationsTableWidget.rowCount()
        self.translationsTableWidget.insertRow(row_position)
        self.translationsTableWidget.setItem(row_position, 0, QtWidgets.QTableWidgetItem(original))
        self.translationsTableWidget.setItem(row_position, 1, QtWidgets.QTableWidgetItem(replacement))

    def edit_translation_in_table(self, row, original, replacement):
        self.translationsTableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(original))
        self.translationsTableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(replacement))

    def edit_mapping_in_table(self, row, original, replacement):
        self.mappingsTableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(original))
        self.mappingsTableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(replacement))

    def on_clear_mappings_clicked(self):
        # Clear all rows in the translations table
        self.mappingsTableWidget.setRowCount(0)

    def on_clear_translations_clicked(self):
        # Clear all rows in the translations table
        self.translationsTableWidget.setRowCount(0)

    def on_apply_mappings_clicked(self):
        # Apply translations from the table to the alternative text
        alternative_text_content = self.textEditAlternativeText.toPlainText()

        # Iterate through all rows in the translations table
        for row in range(self.mappingsTableWidget.rowCount()):
            original_item = self.mappingsTableWidget.item(row, 0)
            replacement_item = self.mappingsTableWidget.item(row, 1)

            if original_item and replacement_item:
                original_text = original_item.text()
                replacement_text = replacement_item.text()

                # Apply the translation
                alternative_text_content = alternative_text_content.replace(original_text, replacement_text)

        # Update the alternative text area with the translated content
        self.textEditAlternativeText.setPlainText(alternative_text_content)

    def on_apply_translations_clicked(self):
        # Apply translations from the table to the alternative text
        alternative_text_content = self.textEditAlternativeText.toPlainText()

        # Iterate through all rows in the translations table
        for row in range(self.translationsTableWidget.rowCount()):
            original_item = self.translationsTableWidget.item(row, 0)
            replacement_item = self.translationsTableWidget.item(row, 1)

            if original_item and replacement_item:
                original_text = original_item.text()
                replacement_text = replacement_item.text()

                # Apply the translation
                alternative_text_content = alternative_text_content.replace(original_text, replacement_text)

        # Update the alternative text area with the translated content
        self.textEditAlternativeText.setPlainText(alternative_text_content)

    def init_preprocessing_combobox(self):
        # Initialize preprocessing combobox with code templates
        self.code_templates = {
            "Template 1": """
def process_text(text):
    ls = []
    for line in text.split("\\n"):
        if line.strip():
            if "START OF REPORT" in line:
                continue
            if "END OF REPORT" in line:
                continue    
            ls.append(line)
    return "\\n".join(ls) 
""",
            "Template 2": "def process_text(text):\n    # Another processing example\n    return text.lower()",
            # Add more templates as needed
        }

        self.preprocessingComboBox.addItems(self.code_templates.keys())
        self.preprocessingComboBox.currentIndexChanged.connect(self.on_template_selection_changed)
        self.on_template_selection_changed(0)  # Initialize editor with the first template

    def on_template_selection_changed(self, index):
        # Update the code editor with the selected template code
        selected_template = self.preprocessingComboBox.currentText()
        self.code_editor.setPlainText(self.code_templates.get(selected_template, ""))

    def on_clear_preprocessing_clicked(self):
        # Clear the code editor
        self.code_editor.clear()

    def on_apply_preprocessing_clicked(self):
        # Apply the code in the editor to the contents of the text areas
        try:
            # Get the code from the editor
            code = self.code_editor.toPlainText()

            # Define a local namespace to safely execute the code
            local_namespace = {}

            # Execute the code in the local namespace
            exec(code, {}, local_namespace)

            # Ensure the required function is defined
            if 'process_text' not in local_namespace:
                raise ValueError("The function 'process_text(text)' must be defined in the code.")

            # Get the function
            process_text = local_namespace['process_text']

            # Get the content from both text areas
            main_text_content = self.textEditMainText.toPlainText()
            alternative_text_content = self.textEditAlternativeText.toPlainText()

            # Apply the function to the contents
            processed_main_text = process_text(main_text_content)
            processed_alternative_text = process_text(alternative_text_content)

            # Display the results in the text areas
            self.textEditMainText.setPlainText(processed_main_text)
            self.textEditAlternativeText.setPlainText(processed_alternative_text)

        except Exception as e:
            error_message = traceback.format_exc()
            QtWidgets.QMessageBox.critical(None, "Error",
                                           f"An error occurred while applying the code:\n\n{error_message}")

    # Existing methods for opening and reloading text files
    def on_open_main_text_clicked(self):
        # Open a file dialog to select a file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Main Text File", "",
                                                             "Text Files (*.txt);;All Files (*)")

        if file_path:
            self.load_text_file(file_path, self.textEditMainText, self.spinBoxMainTextLimit.value())

    def on_open_main_text_clicked(self):
        # Open a file dialog to select a file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Main Text File", "",
                                                             "Text Files (*.txt);;All Files (*)")

        if file_path:
            self.load_text_file(file_path, self.textEditMainText, self.spinBoxMainTextLimit.value())

    def on_reload_main_text_clicked(self):
        # Reload the content of the currently loaded file in textEditMainText
        current_file = self.lineEditMainText.text()
        if current_file:
            self.load_text_file(current_file, self.textEditMainText, self.spinBoxMainTextLimit.value())

    def on_open_alternative_text_clicked(self):
        # Open a file dialog to select a file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Alternative Text File", "",
                                                             "Text Files (*.txt);;All Files (*)")

        if file_path:
            self.load_text_file(file_path, self.textEditAlternativeText, self.spinBoxAlternativeTextLimit.value())

    def on_reload_alternative_text_clicked(self):
        # Reload the content of the currently loaded file in textEditAlternativeText
        current_file = self.lineEditAlternativeText.text()
        if current_file:
            self.load_text_file(current_file, self.textEditAlternativeText, self.spinBoxAlternativeTextLimit.value())

    def load_text_file(self, file_path, text_edit_widget, line_limit):
        """
        Load the content of the file specified by file_path into the given text edit widget.
        Only load the number of lines specified by line_limit.
        If line_limit is 0, load the entire file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                if line_limit > 0:
                    # Read up to line_limit lines
                    lines = []
                    for i, line in enumerate(file):
                        if i >= line_limit:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                else:
                    # Read the entire file
                    content = file.read()

                # Set the file path in the respective QLineEdit
                if text_edit_widget == self.textEditMainText:
                    self.lineEditMainText.setText(file_path)
                else:
                    self.lineEditAlternativeText.setText(file_path)

                # Set the content to the text edit widget
                text_edit_widget.setPlainText(content)

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Could not open file: {e}")

    def on_add_cluster_clicked(self):
        # Create a new cluster using form inputs and add it to the tree
        cluster_data = self.get_cluster_data_from_form()
        if cluster_data:
            cluster_item = QtWidgets.QTreeWidgetItem(self.clusteringTreeWidget)
            cluster_item.setText(0, cluster_data["name"])
            cluster_item.setData(0, QtCore.Qt.UserRole, cluster_data)
            self.clusteringTreeWidget.addTopLevelItem(cluster_item)
            print(f"Cluster '{cluster_data['name']}' added.")

    def on_edit_cluster_clicked(self):
        # Edit the selected cluster in the tree
        selected_item = self.clusteringTreeWidget.currentItem()
        if not selected_item:
            QtWidgets.QMessageBox.warning(self, "Selection Error", "No cluster selected to edit.")
            return

        cluster_data = self.get_cluster_data_from_form()
        if cluster_data:
            selected_item.setText(0, cluster_data["name"])
            selected_item.setData(0, QtCore.Qt.UserRole, cluster_data)
            print(f"Cluster '{cluster_data['name']}' edited.")

    def on_delete_cluster_clicked(self):
        # Delete the selected cluster from the tree
        selected_item = self.clusteringTreeWidget.currentItem()
        if selected_item:
            index = self.clusteringTreeWidget.indexOfTopLevelItem(selected_item)
            self.clusteringTreeWidget.takeTopLevelItem(index)
            print("Selected cluster deleted.")

    def on_tree_item_clicked(self, item):
        # Populate the form with the cluster data when a tree item is clicked
        cluster_data = item.data(0, QtCore.Qt.UserRole)
        if cluster_data:
            self.populate_form_with_cluster_data(cluster_data)
            print(f"Cluster '{cluster_data['name']}' selected, form populated.")

    def on_add_part_clicked(self):
        # Add a new part to the parts list
        dialog = PartDialog(parent=self.MainWindow)
        if dialog.exec_() == QDialog.Accepted:
            part_name, min_val, max_val = dialog.getData()
            item = QListWidgetItem(f"{part_name} (Min: {min_val}, Max: {max_val})")
            item.setData(QtCore.Qt.UserRole, {"name": part_name, "min": min_val, "max": max_val})
            self.clusterPartsListWidget.addItem(item)
            print(f"Part '{part_name}' added.")

    def on_edit_part_clicked(self):
        # Edit the selected part in the parts list
        current_item = self.clusterPartsListWidget.currentItem()
        if current_item:
            part_data = current_item.data(QtCore.Qt.UserRole)
            dialog = PartDialog(part_data["name"], part_data["min"], part_data["max"], parent=self.MainWindow)
            if dialog.exec_() == QDialog.Accepted:
                part_name, min_val, max_val = dialog.getData()
                current_item.setText(f"{part_name} (Min: {min_val}, Max: {max_val})")
                current_item.setData(QtCore.Qt.UserRole, {"name": part_name, "min": min_val, "max": max_val})
                print(f"Part '{part_name}' edited.")

    def on_delete_part_clicked(self):
        # Delete the selected part from the parts list
        current_item = self.clusterPartsListWidget.currentItem()
        if current_item:
            self.clusterPartsListWidget.takeItem(self.clusterPartsListWidget.row(current_item))
            print("Selected part deleted.")

    def on_add_pattern_clicked(self):
        # Add a new part to the parts list
        dialog = PartDialog(parent=self.MainWindow, det="Pattern")
        if dialog.exec_() == QDialog.Accepted:
            part_name, min_val, max_val = dialog.getData()
            item = QListWidgetItem(f"{part_name} (Min: {min_val}, Max: {max_val})")
            item.setData(QtCore.Qt.UserRole, {"name": part_name, "min": min_val, "max": max_val})
            self.clusterPatternsListWidget.addItem(item)
            print(f"Pattern '{part_name}' added.")

    def on_edit_pattern_clicked(self):
        current_item = self.clusterPatternsListWidget.currentItem()
        if current_item:
            part_data = current_item.data(QtCore.Qt.UserRole)
            dialog = PartDialog(part_data["name"], part_data["min"], part_data["max"], parent=self.MainWindow,
                                det="Pattern")
            if dialog.exec_() == QDialog.Accepted:
                part_name, min_val, max_val = dialog.getData()
                current_item.setText(f"{part_name} (Min: {min_val}, Max: {max_val})")
                current_item.setData(QtCore.Qt.UserRole, {"name": part_name, "min": min_val, "max": max_val})
                print(f"Pattern '{part_name}' edited.")

    def on_delete_pattern_clicked(self):
        # Delete the selected pattern from the patterns list
        current_item = self.clusterPatternsListWidget.currentItem()
        if current_item:
            self.clusterPatternsListWidget.takeItem(self.clusterPatternsListWidget.row(current_item))
            print("Selected pattern deleted.")

    def on_clear_clustering_clicked(self):
        # Clear all clusters, parts, patterns, and reset the form fields
        self.clusteringTreeWidget.clear()
        self.clusterPartsListWidget.clear()
        self.clusterPatternsListWidget.clear()
        self.clustering_code_editor.clear()
        self.lineEditClusterName.clear()
        self.lineEditMinLength.clear()
        self.lineEditMaxLength.clear()
        self.lineEditPartsWeight.clear()
        self.lineEditRegexWeight.clear()
        print("All clustering data and form fields cleared.")

    def on_apply_clustering_clicked(self):
        # Apply the clustering logic using the data in the form
        print("Applying clustering logic...")

    def on_export_cluster_clicked(self):
        dc = {
            "preprocessing": self.code_editor.toPlainText(),
            "translations": self.qtablewidget_to_list_of_dicts(self.translationsTableWidget),
            "mappings": self.qtablewidget_to_list_of_dicts(self.mappingsTableWidget),
            "clusters": self.tree_to_dict(self.clusteringTreeWidget)
        }
        self.code_editor2.setPlainText(f"dc = {json.dumps(dc, indent=5)}")

    def on_generate_cluster_clicked(self):
        dc = {
            "preprocessing": self.code_editor.toPlainText(),
            "translations": self.qtablewidget_to_list_of_dicts(self.translationsTableWidget),
            "clusters": self.tree_to_dict(self.clusteringTreeWidget)
        }
        try:
            parser_code = generate_parser_code(dc)
            self.code_editor2.setPlainText(parser_code)
        except Exception as e:
            print(e)

    def execute_code_from_editor(self):
        # Get the code from the code editor
        code = self.code_editor2.toPlainText()
        # Execute the code to define classes dynamically
        local_context = {
            "re": re
        }
        try:
            exec(code, globals(), local_context)
            return local_context.get('GeneratedParser')
        except Exception as e:
            print(f"Error executing code: {e}")
            return None

    def on_run_tagger_clicked(self):
        # Dynamically get the GeneratedParser class
        GeneratedParser = self.execute_code_from_editor()
        if not GeneratedParser:
            return
        # Create an instance of GeneratedParser
        parser = GeneratedParser()
        self.on_reload_main_text_clicked()
        # Get text from textEdit
        document_text = self.textEditMainText.toPlainText()

        # Run tag_document method from GeneratedParser
        tagged_document = parser.get_line_scores(document_text)

        # Function to generate a random light color
        def generate_light_color():
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            return f"rgb({r}, {g}, {b})"

        # Generate HTML content with styled lines
        html_content = "<html><body style='font-family: Arial, sans-serif;'>"
        COLOR_MAP = {tmp: generate_light_color() for tmp in tagged_document.values()}
        for line, tag in tagged_document.items():
            color = COLOR_MAP[tag]  # Generate a random light color for each line
            html_content += f"<div style='background-color: {color}; padding: 5px; margin-bottom: 5px;'>{tag} - {str(line)}</div>"

        html_content += "</body></html>"

        # Set the HTML content back to the QWebEngineView
        self.web_viewMainText.setHtml(html_content)

    def on_run_parser_clicked(self):
        # Dynamically get the GeneratedParser class
        GeneratedParser = self.execute_code_from_editor()
        if not GeneratedParser:
            return

        try:
            # Create an instance of GeneratedParser
            parser = GeneratedParser()
            self.on_reload_main_text_clicked()
            # Get text from textEdit
            document_text = self.textEditMainText.toPlainText()

            # Run get_parsed_lines method from GeneratedParser
            parsed_lines = parser.get_parsed_lines(document_text)

            # Convert parsed_lines to a string to display
            parsed_text = "\n".join([str(line) for line in parsed_lines])

            # Set the text back to the textEdit
            self.textEditMainText.setPlainText(parsed_text)
        except Exception as e:
            print(e)




    def on_run_parser_and_merge_clicked(self):
        # Dynamically get the GeneratedParser class
        GeneratedParser = self.execute_code_from_editor()
        if not GeneratedParser:
            return

        try:
            # Create an instance of GeneratedParser
            parser = GeneratedParser()
            self.on_reload_main_text_clicked()
            # Get text from textEdit
            document_text = self.textEditMainText.toPlainText()

            # Run get_parsed_lines method from GeneratedParser
            parsed_lines = parser.get_parsed_lines(document_text)

            hierarchy = self.tree_to_dict(self.clusteringTreeWidget)
            #hierarchy = [parser.configs()]
            # Convert parsed_lines to a string to display
            parsed_text = "\n".join([str(line) for line in parsed_lines])
            tech = self.techniquesComboBox.currentText()
            if tech == "Up2Down":
                results = parser.process_elements_with_hierarchy(parsed_text, hierarchy, "JENTACULAR")
                results = [s for t in results for s in t]
            else:
                results = parser.process_elements(parsed_text, hierarchy)

            # Set the text back to the textEdit
            self.textEditMainText.setPlainText("\n".join([str(line) for line in results]))
        except Exception as e:
            print(e)

    def get_cluster_data_from_form(self):
        """
        Collect all data from the form to create or edit a cluster.
        """
        try:
            cluster_data = {
                "name": self.lineEditClusterName.text(),
                "min_length": int(self.lineEditMinLength.text()),
                "max_length": int(self.lineEditMaxLength.text()),
                "parts_weight": float(self.lineEditPartsWeight.text()),
                "regex_weight": float(self.lineEditRegexWeight.text()),
                "code": self.clustering_code_editor.toPlainText(),
                "parts": self.get_parts_from_list_widget(self.clusterPartsListWidget),
                "patterns": self.get_patterns_from_list_widget(self.clusterPatternsListWidget)
            }
            return cluster_data
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
            return None

    def populate_form_with_cluster_data(self, cluster_data):
        """
        Populate the form fields with data from a selected cluster.
        """
        self.lineEditClusterName.setText(cluster_data["name"])
        self.lineEditMinLength.setText(str(cluster_data["min_length"]))
        self.lineEditMaxLength.setText(str(cluster_data["max_length"]))
        self.lineEditPartsWeight.setText(str(cluster_data["parts_weight"]))
        self.lineEditRegexWeight.setText(str(cluster_data["regex_weight"]))
        self.clustering_code_editor.setPlainText(cluster_data["code"])

        self.clusterPartsListWidget.clear()
        for part in cluster_data["parts"]:
            item = QListWidgetItem(f"{part['name']} (Min: {part['min']}, Max: {part['max']})")
            item.setData(QtCore.Qt.UserRole, part)
            self.clusterPartsListWidget.addItem(item)

        self.clusterPatternsListWidget.clear()
        for pattern in cluster_data["patterns"]:
            self.clusterPatternsListWidget.addItem(pattern)

    def get_parts_from_list_widget(self, list_widget):
        # Retrieve parts from the list widget
        parts = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            part_data = item.data(QtCore.Qt.UserRole)
            parts.append(part_data)
        return parts

    def get_patterns_from_list_widget(self, list_widget):
        # Retrieve parts from the list widget
        parts = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            part_data = item.data(QtCore.Qt.UserRole)
            parts.append(part_data)
        return parts


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ParserApp()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
