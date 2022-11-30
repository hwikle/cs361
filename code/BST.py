from typing import List


class BSTNode:
    def __init__(self, key: int, left: 'BSTNode' = None, right: 'BSTNode' = None, parent: 'BSTNode' = None):
        self.parent = parent
        self.left = left
        self.right = right
        self.key = key

    def __eq__(self, other: 'BSTNode') -> bool:
        return self.key == other.key

    def __lt__(self, other: 'BSTNode') -> bool:
        return self.key < other.key

    def __gt__(self, other: 'BSTNode') -> bool:
        return self.key > other.key


class BST:
    def __init__(self, root: BSTNode = None):
        self.root = root


def transplant(T: BST, u: BSTNode, v: BSTNode) -> None:
    if u.parent is None:
        T.root = v
    elif u is u.parent.left:
        u.parent.left = v
    else:
        u.parent.right = v
    if v is not None:
        v.parent = u.parent


def minimum(x: BSTNode) -> BSTNode:
    while x.left is not None:
        x = x.left

    return x


def insert(T: BST, z: BSTNode) -> None:
    x = T.root
    y = None

    while x is not None:
        y = x

        if z < x:
            x = x.left
        else:
            x = x.right

    z.parent = y

    if y is None:
        T.root = z
    elif z < y:
        y.left = z
    else:
        y.right = z


def delete(T: BST, z: BSTNode) -> None:
    if z.left is None:
        transplant(T, z, z.right)
    elif z.right is None:
        transplant(T, z, z.left)
    else:
        y = minimum(z.right)

        if y.parent is not z:
            transplant(T, z, y)
            y.right = z.right
            y.right.parent = y

        transplant(T, z, y)
        y.left = z.left
        y.left.parent = y


def inorder_walk(x: BSTNode) -> List[BSTNode]:
    if x is not None:
        return inorder_walk(x.left) + [x] + inorder_walk(x.right)

    return []


def tree_equal(u: BSTNode, v: BSTNode):
    if u is None and v is None:
        return True
    elif u is None or v is None or u != v:
        return False
    else:
        return tree_equal(u.left, v.left) and tree_equal(u.right, v.right)