class NullTarget:
    """Placeholder target that returns None when instantiated by Hydra.
    
    Used for optional components like schedulers. When _target_ is set to this
    class, Hydra's instantiate() will return None, allowing clean handling of
    optional components without conditionals in train.py.
    """
    def __new__(cls, *args, **kwargs):
        return None
