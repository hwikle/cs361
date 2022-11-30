from BST import BST, BSTNode


class AVLNode(BSTNode):
    def __init__(self, key: int, **kwargs) -> None:
        super().__init__(key, **kwargs)
        self.height = 0


class AVLTree(BST):
    def __init__(self, root: AVLNode):
        super().__init__(root)


def is_balanced(T: BST):
    ...
