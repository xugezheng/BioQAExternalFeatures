import requests
import nltk
import json
import tokenization
import tqdm
import time


def query_raw(text, url="https://bern.korea.ac.kr/plain", f={}):
    results = " "
    while results == " ":
        try:
            results = requests.post(url, data={'sample_text': text}).json()
        except:
            with open("rawBioner_" + "error.json", 'w') as outf:
                json.dump(f, outf, indent=4)
                print(text)
                print('Error File has been written')
            time.sleep(30)
            continue
    return results


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def sent2token(sent):
    tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in sent:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(tokens) - 1)
    return tokens, char_to_word_offset


def fetch_bioner(filename, mode='d'):
    with open(filename) as f:
        file = json.load(f)
    for paras in file['data']:
        for ctxt in tqdm.tqdm(paras['paragraphs']):
            paragraph_text = ctxt["context"]
            context_re = query_raw(paragraph_text, f=file)
            ctxt["context_raw_bioner"] = context_re
            for qa_pair in ctxt['qas']:
                q = qa_pair["question"]
                question_re = query_raw(q)
                qa_pair["question_raw_bioner"] = question_re
    if mode == "f":
        output_filename = filename.split('/')[-1]
        with open("rawBioner_" + output_filename, 'w') as outf:
            json.dump(file, outf, indent=4)
            print('Raw File has been written')
            return {}
    else:
        return file


def align_bioner_tokens(filename, sample=False):
    count = 0
    #vars
    tags_dict = {"disease": 1, "gene": 2, "species": 3, "drug": 4, "mutation": 5, "pathway": 6, "miRNA":7}
    vocab_file = "vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)

    with open(filename) as f:
        file = json.load(f)
    for paras in file['data']:
        count += 1
        for ctxt in tqdm.tqdm(paras['paragraphs']):

            # for context
            paragraph_text = ctxt["context"]
            doc_tokens, char_to_tokens = sent2token(paragraph_text)
            c_bioner_re = ctxt["context_raw_bioner"]
            bioner_ids = [0] * len(doc_tokens)
            if 'denotations' not in c_bioner_re.keys():
                print(paragraph_text)
                print(count)
            else:
                if len(c_bioner_re['denotations']) > 0:
                    for ner in c_bioner_re['denotations']:
                        char_start = ner['span']['begin']
                        char_end = ner['span']['end']
                        obj = ner['obj']
                        for index in range(char_to_tokens[char_start],char_to_tokens[char_end] + 1):
                            bioner_ids[index] = tags_dict[obj]
                    if sample:
                        ctxt["context_raw_bioner"] = []
            assert len(bioner_ids) == len(doc_tokens)
            ctxt["context_bioner_ids"] = bioner_ids
            # for question
            for qa_pair in ctxt['qas']:
                q = qa_pair["question"]
                q_bioner_re = qa_pair["question_raw_bioner"]
                if sample:
                    qa_pair["question_raw_bioner"] = []
                q_word = tokenizer.tokenize(q)
                question_bioner_ids = [0] * len(q_word)
                if len(q_bioner_re["denotations"]) > 0:
                    for q_ner in q_bioner_re["denotations"]:
                        q_char_start = q_ner['span']['begin']
                        q_char_end = q_ner['span']['end']
                        q_obj = q_ner['obj']
                        ner_text = q[q_char_start:q_char_end]
                        if ner_text in q_word:
                            question_bioner_ids[q_word.index(ner_text)] = tags_dict[q_obj]
                        else:
                            tokenized_ner = tokenizer.tokenize(ner_text)
                            try:
                                indice = q_word.index(tokenized_ner[0])
                            except:
                                print(qa_pair['id'])
                                print(tokenized_ner)
                                print(q_word)
                                continue
                            while_count = 0
                            while len(tokenized_ner) >= 2 and q_word[indice + 1] != tokenized_ner[1] and while_count <= 10:
                                while_count += 1
                                indice += q_word[indice + 1:].index(tokenized_ner[0]) + 1
                            for i in range(len(tokenized_ner)):
                                question_bioner_ids[indice + i] = tags_dict[q_obj]
                assert len(question_bioner_ids) == len(q_word)
                qa_pair["question_bioner_ids"] = question_bioner_ids

    output_filename = filename  # .split('/')[1]
    if sample:
        output_filename = 'sample_' + output_filename
    with open("bioner_" + output_filename, 'w') as outf:
        json.dump(file, outf, indent=4)
        print('File has been written')

    return


if __name__ == '__main__':
    # text = "Tendon protein synthesis rate in classic Ehlers-Danlos patients can be stimulated with insulin-like growth factor-I. The classic form of Ehlers-Danlos syndrome (cEDS) is an inherited connective tissue disorder, where mutations in type V collagen-encoding genes result in abnormal collagen fibrils. Thus the cEDS patients have pathological connective tissue morphology and low stiffness, but the rate of connective tissue protein turnover is unknown. We investigated whether cEDS affected the protein synthesis rate in skin and tendon, and whether this could be stimulated in tendon tissue with insulin-like growth factor-I (IGF-I). Five patients with cEDS and 10 healthy, matched controls (CTRL) were included. One patellar tendon of each participant was injected with 0.1 ml IGF-I (Increlex, Ipsen, 10 mg/ml) and the contralateral tendon with 0.1 ml isotonic saline as control. The injections were performed at both 24 and 6 h prior to tissue sampling. The fractional synthesis rate (FSR) of proteins in skin and tendon was measured with the stable isotope technique using a flood-primed continuous infusion over 6 h. After the infusion one skin biopsy and two tendon biopsies (one from each patellar tendon) were obtained. We found similar baseline FSR values in skin and tendon in the cEDS patients and controls [skin: 0.005 \u00b1 0.002 (cEDS) and 0.007 \u00b1 0.002 (CTRL); tendon: 0.008 \u00b1 0.001 (cEDS) and 0.009 \u00b1 0.002 (CTRL) %/h, mean \u00b1 SE]. IGF-I injections significantly increased FSR values in cEDS patients but not in controls (delta values: cEDS 0.007 \u00b1 0.002, CTRL 0.001 \u00b1 0.001%/h). In conclusion, baseline protein synthesis rates in connective tissue appeared normal in cEDS patients, and the patients responded with an increased tendon protein synthesis rate to IGF-I injections."
    # tokens, char_to_tokens = sent2token(text)
    # re = query_raw(text)
    # print(re)
    # for ner in re['denotations']:
    #     char_start = ner['span']['begin']
    #     char_end = ner['span']['end']
    #     print("------")
    #     print(tokens[char_to_tokens[char_start]:char_to_tokens[char_end]+1])
    #     print(ner['obj'])
    # #print(text[16:45])
    # print("Finished")
    fetch_bioner("all_json/6b.json", mode='f')
