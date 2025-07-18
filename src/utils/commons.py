def get_all_subclasses(cls):
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(get_all_subclasses(subclass))
    return subclasses


def get_classes_from_module(module_startswith, parent_class):
    return {
        cls.__name__: cls
        for cls in get_all_subclasses(parent_class)
        if cls.__module__.startswith(module_startswith)
    }


def get_resize_transform(size):
    from torchvision import transforms as T

    return T.Compose([T.Resize((size, size)), T.ToTensor()])
