class Token:
    def __init__(self, raw:str, pos:str) -> None:
        self.raw = raw
        self.pos = pos
    def __repr__(self) -> str:
        return "{raw}({pos})".format(raw=self.raw, pos=self.pos)

        