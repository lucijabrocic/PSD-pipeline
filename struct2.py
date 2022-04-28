from spacy.tokens import Span

class IOBType(str):

    @property
    def iob(self):
        i = self.find("-")
        if i >= 0:
            return self[:i]
        return self

    @property
    def type_(self):
        i = self.find("-")
        if i >= 0:
            return self[i + 1:]
        return ""

    @staticmethod
    def create(iob, type_=""):
        if type_ != "":
            return IOBType(f"{iob}-{type_}")
        return IOBType(iob)

    @staticmethod
    def begin(lbl):
        return IOBType(f"B-{lbl}")

    @staticmethod
    def inside(lbl):
        return IOBType(f"I-{lbl}")

    @staticmethod
    def other():
        return IOBType("O")


class LabelSpan(Span):

    def __repr__(self):
        return f"{self.label_}: {Span.__repr__(self)}"


