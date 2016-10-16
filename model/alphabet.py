class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id

    def add(self, item):
        idx, freq = self.get(item, (None, None))

        if idx is None:
            idx = self.fid
            self[item] = (idx, 1)
            self.fid += 1
        else:
            self[item] = (idx, freq + 1)

        return idx
