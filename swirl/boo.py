import re
from string import digits


class BagOfOperators(object):
    def __init__(self):
        self.replacings = [(" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""), ("::text", "")]
        self.remove_digits = str.maketrans("", "", digits)
        self.INTERESTING_OPERATORS = [
            "Seq Scan",
            "Hash Join",
            "Nested Loop",
            "CTE Scan",
            "Index Only Scan",
            "Index Scan",
            "Merge Join",
            "Sort",
        ]

        self.relevant_operators = None

    def boo_from_plan(self, plan):
        self.relevant_operators = []
        self._parse_plan(plan)

        return self.relevant_operators

    def _parse_plan(self, plan):
        node_type = plan["Node Type"]

        if node_type in self.INTERESTING_OPERATORS:
            node_representation = self._parse_node(plan)
            self.relevant_operators.append(node_representation)
        if "Plans" not in plan:
            return
        for sub_plan in plan["Plans"]:
            self._parse_plan(sub_plan)

    def _stringify_attribute_columns(self, node, attribute):
        attribute_representation = f"{attribute.replace(' ', '')}_"
        if attribute not in node:
            return attribute_representation

        value = node[attribute]

        for replacee, replacement in self.replacings:
            value = value.replace(replacee, replacement)

        value = re.sub('".*?"', "", value)
        value = re.sub("'.*?'", "", value)
        value = value.translate(self.remove_digits)

        return value

    def _stringify_list_attribute(self, node, attribute):
        attribute_representation = f"{attribute.replace(' ', '')}_"
        if attribute not in node:
            return attribute_representation

        assert isinstance(node[attribute], list)
        value = node[attribute]

        for element in value:
            attribute_representation += f"{element}_"

        return attribute_representation

    def _parse_bool_attribute(self, node, attribute):
        attribute_representation = f"{attribute.replace(' ', '')}_"

        if attribute not in node:
            return attribute_representation

        value = node[attribute]
        attribute_representation += f"{value}_"

        return attribute_representation

    def _parse_string_attribute(self, node, attribute):
        attribute_representation = f"{attribute.replace(' ', '')}_"

        if attribute not in node:
            return attribute_representation

        value = node[attribute]
        attribute_representation += f"{value}_"

        return attribute_representation

    def _parse_seq_scan(self, node):
        assert "Relation Name" in node

        node_representation = ""
        node_representation += f"{node['Relation Name']}_"

        node_representation += self._stringify_attribute_columns(node, "Filter")

        return node_representation

    def _parse_index_scan(self, node):
        assert "Relation Name" in node

        node_representation = ""
        node_representation += f"{node['Relation Name']}_"

        node_representation += self._stringify_attribute_columns(node, "Filter")
        node_representation += self._stringify_attribute_columns(node, "Index Cond")

        return node_representation

    def _parse_index_only_scan(self, node):
        assert "Relation Name" in node

        node_representation = ""
        node_representation += f"{node['Relation Name']}_"

        node_representation += self._stringify_attribute_columns(node, "Index Cond")

        return node_representation

    def _parse_cte_scan(self, node):
        assert "CTE Name" in node

        node_representation = ""
        node_representation += f"{node['CTE Name']}_"

        node_representation += self._stringify_attribute_columns(node, "Filter")

        return node_representation

    def _parse_nested_loop(self, node):
        node_representation = ""

        node_representation += self._stringify_attribute_columns(node, "Join Filter")

        return node_representation

    def _parse_hash_join(self, node):
        node_representation = ""

        node_representation += self._stringify_attribute_columns(node, "Join Filter")
        node_representation += self._stringify_attribute_columns(node, "Hash Cond")

        return node_representation

    def _parse_merge_join(self, node):
        node_representation = ""

        node_representation += self._stringify_attribute_columns(node, "Merge Cond")

        return node_representation

    def _parse_sort(self, node):
        node_representation = ""

        node_representation += self._stringify_list_attribute(node, "Sort Key")

        return node_representation

    def _parse_node(self, node):
        node_representation = f"{node['Node Type'].replace(' ', '')}_"

        if node["Node Type"] == "Seq Scan":
            node_representation += f"{self._parse_seq_scan(node)}"
        elif node["Node Type"] == "Index Only Scan":
            node_representation += f"{self._parse_index_only_scan(node)}"
        elif node["Node Type"] == "Index Scan":
            node_representation += f"{self._parse_index_scan(node)}"
        elif node["Node Type"] == "CTE Scan":
            node_representation += f"{self._parse_cte_scan(node)}"
        elif node["Node Type"] == "Nested Loop":
            node_representation += f"{self._parse_nested_loop(node)}"
        elif node["Node Type"] == "Hash Join":
            node_representation += f"{self._parse_hash_join(node)}"
        elif node["Node Type"] == "Merge Join":
            node_representation += f"{self._parse_merge_join(node)}"
        elif node["Node Type"] == "Sort":
            node_representation += f"{self._parse_sort(node)}"
        else:
            raise ValueError("_parse_node called with unsupported Node Type.")

        return node_representation
