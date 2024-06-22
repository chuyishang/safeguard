class DotDict(dict):
    """
    Dictionary subclass that supports dot notation access.
    You can wrap any dictionary using DotDict instead.
    """
    def __init__(self, data):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            else:
                self[key] = value

    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value