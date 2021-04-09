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
