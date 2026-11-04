"""用 tree-sitter 解析 C 源码，抽出所有函数。

用法：
    python parse_c.py sample.c

输出每个函数的：名称、起止行号、签名、完整源码。
这是后面所有工作的地基——先能把函数切出来，才谈得上检索。
"""

import sys

import tree_sitter_c
from tree_sitter import Language, Parser

LANG = Language(tree_sitter_c.language())


def make_parser():
    """兼容 tree-sitter 新旧两套 API。"""
    try:
        return Parser(LANG)          # tree-sitter >= 0.22
    except TypeError:
        p = Parser()
        p.set_language(LANG)         # 旧版本
        return p


def walk(node):
    """深度优先遍历语法树，产出所有 function_definition 节点。"""
    if node.type == "function_definition":
        yield node
    for child in node.children:
        yield from walk(child)


def get_name(func_node):
    """从 declarator 里挖出函数名。

    C 的声明可能层层包裹：
        foo()            → declarator: identifier
        *foo()           → declarator: pointer_declarator → identifier
        (*foo)(int)      → 更复杂
    所以一路往下找 identifier。
    """
    node = func_node.child_by_field_name("declarator")
    while node is not None:
        if node.type == "identifier":
            return node.text.decode()
        node = node.child_by_field_name("declarator")
    return "<unknown>"


def get_signature(func_node):
    """函数签名 = 从节点开头到函数体之前的那些字。"""
    body = func_node.child_by_field_name("body")
    if body is None:
        return func_node.text.decode().strip()
    head = func_node.text[: body.start_byte - func_node.start_byte]
    return head.decode().strip().replace("\n", " ")


def parse_file(path):
    """解析一个 C 文件，返回函数列表。"""
    source = open(path, "rb").read()
    tree = make_parser().parse(source)

    funcs = []
    for node in walk(tree.root_node):
        funcs.append({
            "file": path,
            "name": get_name(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "signature": get_signature(node),
            "code": node.text.decode(),
        })
    return funcs


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.c"
    funcs = parse_file(path)

    print(f"{path} 里共找到 {len(funcs)} 个函数：\n")
    print(f"{'函数名':<14}{'行号':<12}签名")
    print("-" * 72)
    for f in funcs:
        span = f"{f['start_line']}-{f['end_line']}"
        sig = f["signature"][:40]
        print(f"{f['name']:<14}{span:<12}{sig}")

    print("\n第一个函数的完整源码：")
    print("-" * 72)
    print(funcs[0]["code"])
