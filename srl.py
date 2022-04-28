from spacy.language import Language
from spacy.tokens import Token, Doc, Span

from struct2 import IOBType, LabelSpan



def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            # We treat auxiliaries as verbs only for English for now to be safe. We didn't want to
            # hypothetically break the predictor for unknown number of other languages where
            # auxiliaries can't be treated this way.
            if word.pos_ in ["VERB", "NOUN"] or (self._language.startswith("en_") and word.pos_ == "AUX"):
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

from allennlp_models.structured_prediction.predictors import SemanticRoleLabelerPredictor
SemanticRoleLabelerPredictor.tokens_to_instances = tokens_to_instances 





class SrlComponent(object):
    name = "srl"

    labels = ('ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARG6',
              'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR',
              'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC',
              'ARGM-MNR', 'ARGM-MOD', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP',
              'ARGM-REC', 'ARGM-TMP', 'V')

    @staticmethod
    def create_srl_span(doc, start, end, label):
        return LabelSpan(doc, start, end, label)

    @staticmethod
    def doc_srl_getter(doc):
        doc_srl = {}
        for sent in doc.sents:
            doc_srl.update(sent._.srl)
        return doc_srl

    @staticmethod
    def tok_srl_getter(tok):
        tok_srl = {}
        for verb in tok.sent._.srl:
            if tok == verb:
                tok_srl[verb] = IOBType.create("B", "V")
            else:
                for arg in tok.sent._.srl[verb]:
                    if tok.i == arg[0].i:
                        tok_srl[verb] = IOBType.begin(arg.label_)
                        break
                    elif arg[0].i < tok.i <= arg[-1].i:
                        tok_srl[verb] = IOBType.inside(arg.label_)
                        break
                if verb not in tok_srl:
                    tok_srl[verb] = IOBType.other()
        return tok_srl

    def __init__(self, nlp: Language, model="structured-prediction-srl-bert"):
        if isinstance(model, str):
            import allennlp_models.pretrained
            self.srl_model = allennlp_models.pretrained.load_predictor(model)
        else:
            self.srl_model = model

        self.register()


    @staticmethod
    def register():
        # Register custom extension on the Token, Span, Doc
        Token.set_extension("srl", getter=SrlComponent.tok_srl_getter, force=True)
        Span.set_extension("srl", default={}, force=True)
        Doc.set_extension("srl", getter=SrlComponent.doc_srl_getter, force=True)


    def set_annotation(self, sent, sent_tags):
        for tags in sent_tags:
            verb, args = None, []
            start = label = None
            for i, tag in enumerate(tags + ["O"]):
                if tag.startswith(("B", "O")) and start is not None:
                    if label == "V":
                        verb = sent[start]
                        args.append(self.create_srl_span(sent.doc, verb.i, verb.i + 1, "V"))
                    else:
                        offset = i - start
                        args.append(self.create_srl_span(sent.doc, sent[start].i, sent[start].i + offset, label))
                    start = None
                if tag.startswith("B") and start is None:
                    start = i
                    label = tag[2:]

            if verb is not None:
                sent._.srl[verb] = args






    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            srl_anno = self.srl_model.predict_tokenized([t.text for t in sent])
            #print(srl_anno)
            ### nouns and light verbs ###
            #####################################
            def get_verb_index(i, verbs):
                for vi, v in enumerate(verbs):
                    if "B-V" not in v["tags"]: continue
                    ind=v["tags"].index("B-V")
                    if i==ind:
                        return vi

            for i, verb in enumerate(srl_anno["verbs"]):
                if "B-V" in verb["tags"]:
                    verb_tok_ind=verb["tags"].index("B-V")
                    if sent[verb_tok_ind].pos_ =="NOUN" and not(sent[verb_tok_ind].text.endswith("ing")) and "B-ARGM-LVB" not in verb["tags"]:
                        verb["tags"]=["O"]*len(srl_anno["words"])
                    
                    if "B-ARGM-LVB" in verb["tags"]: 
                        lv_ind=verb["tags"].index("B-ARGM-LVB")
                        lv_verb_ind=get_verb_index(lv_ind, srl_anno["verbs"])
                        srl_anno["verbs"][lv_verb_ind]["tags"]=["O"]*len(srl_anno["words"])
                        srl_anno["verbs"][lv_verb_ind]["tags"][verb_tok_ind]="B-ARGM-PRR"
                        srl_anno["verbs"][lv_verb_ind]["tags"][lv_ind]="B-V"
            ####################################

            sent_tags = [item["tags"] for item in srl_anno["verbs"]]
            self.set_annotation(sent, sent_tags)
        return doc


def add_to_pipe(nlp):
    # srl_component = SrlComponent(nlp)
    @Language.factory("srl_component", assigns=["doc._.srl", "span._.srl", "token._.srl"])
    def create_srl_component(nlp, name):
        return SrlComponent(nlp)

    if "srl" in nlp.pipe_names:
        nlp.replace_pipe(component=srl_component, name="srl")
    else:
        # nlp.add_pipe(srl_component, name="srl", last=True)
        nlp.add_pipe(factory_name="srl_component", last=True)

    return nlp

