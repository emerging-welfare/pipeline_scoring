from pynlpl.formats import folia
import sys
from glob import glob
import pandas as pd
import re
from conlleval import evaluate

class_list = ["fname", "place", "etime"]

trigger_list = ["etype", "emention"]

other_class_list = ["pname", "name", "type"]

semantic_type_list = ["demonst", "ind_act", "group_clash", "arm_mil", "elec_pol", "other"]

pred_class_list = ["etime", "fname", "participant", "organizer", "target", "trigger", "place"]

def flatten(x):
    return [b for a in x for b in a]

def sorted_nicely(l):
        # Copied from this post -> https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
        def convert(text): return int(text) if text.isdigit() else text
        def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

        return sorted(l, key=alphanum_key)

def cascade_doc_to_token(row, source_num, target_num):

    token_labels = row["pred_token_labels" + str(source_num)]
    if row.pred_doc_label == 1:
        new_token_labels = token_labels
    else:
        new_token_labels = []
        for labels in token_labels:
            new_token_labels.append(["O"] * len(labels))

    row["pred_token_labels" + str(target_num)] = new_token_labels

    return row

def cascade_sent_to_token(row, source_num, target_num):

    new_token_labels = []
    for sent_label, token_labels in zip(row.pred_sent_labels, row["pred_token_labels" + str(source_num)]):
        if sent_label == 0:
            new_labels = ["O"] * len(token_labels)
        else:
            new_labels = token_labels

        new_token_labels.append(new_labels)

    row["pred_token_labels" + str(target_num)] = new_token_labels
    return row

def BIO_to_sents(token_list, label_list):
    token_labels = []
    all_tokens = []
    tokens = []
    labels = []
    prev_token_label = "O"
    for token,label in zip(token_list, label_list):
        if token == "[SEP]":
            token_labels.append(labels)
            all_tokens.append(tokens)
            tokens = []
            labels = []
        elif token == "SAMPLE_START":
            continue
        elif token == "":
            continue
        else:
            if label.startswith("I-") and (prev_token_label == "O" or prev_token_label[2:] != label[2:]):
                label = "B-" + label[2:]

            labels.append(label)
            prev_token_label = label
            tokens.append(token)

    token_labels.append(labels) # add last sentence
    all_tokens.append(tokens)
    return token_labels, all_tokens

def from_json_output(json_file):
    preds = pd.read_json(json_file, orient="records", lines=True)

    def to_token_labels(row):
        row["token_labels"], _ = BIO_to_sents(row.tokens, row.token_labels)
        return row

    preds = preds.apply(to_token_labels, axis=1)

    return preds

def folia_to_BIO(folia_doc, folia_doc_label, gold=False):

    length_of_doc = sum([1 for paragraph in folia_doc.paragraphs() for sent in paragraph.sentences()])
    folia_doc_sent_labels = []
    j = 0
    sent_df = pd.DataFrame(columns = ["word", "id", "cls"])
    stop = False
    for paragraph in folia_doc.paragraphs():
        for sentence in paragraph.sentences():
            j += 1

            for word in sentence.words():
                sent_df = sent_df.append({"word":word.text(), "id":word.id, "cls":"O"}, ignore_index=True)

            if folia_doc_label == 1:
                if gold:
                    folia_doc_sent_labels.append(int(folia_doc.metadata["Sentence"+str(j)]))
                else:
                    folia_doc_sent_labels.append(int(sentence.cls))

                for layer in sentence.select(folia.EntitiesLayer):
                    for entity in layer.select(folia.Entity):

                        if gold:
                        # Get Everything
                            if entity.cls in class_list:
                                cls = entity.cls
                            elif entity.cls == "loc": # QUESTION: Might want to add to class_list
                                cls = "fname"
                            elif entity.cls in trigger_list:
                                cls = "trigger"
                            # elif entity.cls in semantic_type_list:
                            #     cls = entity.cls
                            elif entity.cls in other_class_list:
                                if "Target" in entity.set:
                                    cls = "target"
                                elif "Organizer" in entity.set:
                                    cls = "organizer"
                                elif "Participant" in entity.set:
                                    cls = "participant"
                                else:
                                    # print("Something wrong!!!")
                                    pass
                            else:
                                continue
                        else: # prediction
                            cls = entity.cls
                            if cls not in pred_class_list:
                                continue

                        curr_words = sorted_nicely([word.id for word in entity.wrefs()])
                        for h, word in enumerate(curr_words):

                            if sent_df[sent_df.id == word].cls.iloc[0] != "O":
                                if cls in sent_df[sent_df.id == word].cls.iloc[0]:
                                    # same_overlap += 1
                                    # If this is not Beginning of annotation and the current tag starts with "B-". Makes these two annotations continuous.
                                    # If h != 0, means that for this annot we already tagged with B-, so we need change B- in order to make this continuous.
                                    print("Same Overlapping")
                                    # print(folder)
                                    print("Batch + filename : ", filename)
                                    print("Annotation's text : ", entity.text())
                                    print("The word : ", gold_doc[word].text())
                                    print("Annotation 1 tag : ", sent_df[sent_df.id == word].cls.iloc[0][2:])
                                    print("Annotation 2 tag : ", cls)
                                    # print(word)
                                    print("-----")
                                    if h != 0 and "B-" in sent_df[sent_df.id == word].cls.iloc[0]:
                                        sent_df.loc[sent_df.id == word, ["cls"]] = "I-" + cls
                                        # print("Hey")
                                        # print([w.id for w in entity.wrefs()])
                                        # print(sent_df[sent_df.id == word].id.iloc[0])
                                        # print(cls)
                                        # print("-----------------")

                                    continue

                                stop = True
                                print("Overlapping")
                                print("Batch + filename : ", filename)
                                print("Annotation's text : ", entity.text())
                                print("The word : ", gold_doc[word].text())
                                print("Annotation 1 tag : ", sent_df[sent_df.id == word].cls.iloc[0][2:])
                                print("Annotation 2 tag : ", cls)
                                print("-----")

                                continue

                            if h == 0:
                                sent_df.loc[sent_df.id == word, ["cls"]] = "B-" + cls
                            else:
                                sent_df.loc[sent_df.id == word, ["cls"]] = "I-" + cls

            if j < length_of_doc - 1:
                sent_df = sent_df.append({"word":"[SEP]", "id":"", "cls":"O"}, ignore_index=True)

#    if len(sent_df) > 2 and not stop:

    token_labels, gold_tokens = BIO_to_sents(sent_df.word.tolist(), sent_df.cls.tolist())

    if folia_doc_label == 1:
        # sent_df = sent_df.iloc[0:-3]

        return folia_doc_sent_labels, token_labels, gold_tokens
    else:
        return [0] * length_of_doc, token_labels, gold_tokens


# Provide full paths
gold_folder = sys.argv[1]
pred_folder = sys.argv[2] # or pred json file if not FOLIA
FOLIA = False

all_df = pd.DataFrame(columns=["filename", "gold_doc_label", "gold_sent_labels", "gold_tokens", "gold_token_labels", "pred_doc_label", "pred_sent_labels", "pred_token_labels1"])

if not FOLIA:
    preds = from_json_output(pred_folder)

for filename in glob(gold_folder + "/*.folia.xml"):
    gold_doc = folia.Document(file=filename)
    filename = re.sub(r"^.*\/([^\/]*)$", r"\g<1>", filename)

    try:
        if gold_doc.metadata["RelevantCountry"] == "No":
            continue
    except:
        pass

    gold_doc_label = 1 if gold_doc.metadata["Event"] == "Yes" else 0
    # gold_doc_violent = gold_doc["Violent"]

    gold_sent_labels, gold_token_labels, gold_tokens = folia_to_BIO(gold_doc, gold_doc_label, gold=True)

    if FOLIA:
        pred_file = pred_folder + "/" + filename
        pred_doc = folia.Document(file=pred_file)
        try:
            pred_doc_label = int(pred_doc.metadata["Document_label"])
        except:
            pred_doc_label = 0
            # pred_sent_labels = [0] * len([sent for para in doc.paragraphs() for sent in para.sentences()])

        pred_sent_labels, pred_token_labels, _ = folia_to_BIO(pred_doc, pred_doc_label)
    else:
        # new_filename = re.sub(r"-h6j7k8-", r"%", filename)
        # new_filename = re.sub(r"\.folia\.xml$", r".ece", new_filename)
        new_filename = gold_doc.metadata["filename"]
        el = preds[preds.id == new_filename].iloc[0]

        pred_doc_label = el.doc_label
        pred_sent_labels, pred_token_labels = el.sent_labels, el.token_labels

    all_df = all_df.append({"filename":filename, "gold_doc_label":gold_doc_label, "gold_sent_labels":gold_sent_labels, "gold_tokens":gold_tokens, "gold_token_labels":gold_token_labels, "pred_doc_label":pred_doc_label, "pred_sent_labels":pred_sent_labels, "pred_token_labels1":pred_token_labels}, ignore_index=True)


# cascade document labels to token
all_df = all_df.apply(lambda x:cascade_doc_to_token(x, 1, 3), axis=1)

# cascade sentence labels to token
all_df = all_df.apply(lambda x:cascade_sent_to_token(x, 1, 2), axis=1)
all_df = all_df.apply(lambda x:cascade_sent_to_token(x, 3, 4), axis=1)

all_df.gold_tokens = all_df.gold_tokens.apply(flatten)
all_df.gold_token_labels = all_df.gold_token_labels.apply(flatten)
all_df.pred_token_labels1 = all_df.pred_token_labels1.apply(flatten)
all_df.pred_token_labels2 = all_df.pred_token_labels2.apply(flatten)
all_df.pred_token_labels3 = all_df.pred_token_labels3.apply(flatten)
all_df.pred_token_labels4 = all_df.pred_token_labels4.apply(flatten)

# For error analysis
all_df[["filename", "gold_doc_label", "pred_doc_label", "gold_sent_labels", "pred_sent_labels", "gold_tokens", "gold_token_labels", "pred_token_labels1", "pred_token_labels2", "pred_token_labels3", "pred_token_labels4"]].to_json("all_df.json", orient="records", lines=True, force_ascii=False)

print(len(all_df))

print("Token error : ")
precision, recall, f1 = evaluate(flatten(all_df.gold_token_labels.tolist()), flatten(all_df.pred_token_labels1.tolist()))
print("Weighted Average Precision : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

print("\n\n")
print("Sentence + Token error : ")
precision, recall, f1 = evaluate(flatten(all_df.gold_token_labels.tolist()), flatten(all_df.pred_token_labels2.tolist()))
print("Weighted Average Precision : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

print("\n\n")
print("Document + Token cascaded error : ")
precision, recall, f1 = evaluate(flatten(all_df.gold_token_labels.tolist()), flatten(all_df.pred_token_labels3.tolist()))
print("Weighted Average Precision : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

print("\n\n")
print("Document + Sentence + Token cascaded error : ")
precision, recall, f1 = evaluate(flatten(all_df.gold_token_labels.tolist()), flatten(all_df.pred_token_labels4.tolist()))
print("Average Precision : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))
