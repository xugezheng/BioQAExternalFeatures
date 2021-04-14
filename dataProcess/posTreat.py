import nltk
import json
import tokenization


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def sent2token(sent):
    tokens = []
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
    return tokens


def tag_alignment(doc_tokens, tag_tokens):
    tag_tokens_flat = []
    final_tag_list = []
    for i in tag_tokens:
        tag_tokens_flat += i
    offset = 0
    if len(doc_tokens) == len(tag_tokens_flat):
        for i in tag_tokens_flat:
            final_tag_list.append(i[1])
    else:
        for index, t in enumerate(doc_tokens):
            tag_index = index + offset
            target_token = tag_tokens_flat[tag_index][0]
            while t.replace("\u3000", "") != target_token:
                offset += 1
                target_token += tag_tokens_flat[index + offset][0]
            final_tag_list.append(tag_tokens_flat[tag_index][1])
        assert len(doc_tokens) == len(final_tag_list)
    return final_tag_list


def main_treat(filename):
    with open(filename) as f:
        file = json.load(f)
    vocab_file = "vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    count = 0
    error_num = 0
    tag_dict = {}
    tags = [
        "IN",
        "DT",
        "NN",
        "VBZ",
        "NNP",
        "JJ",
        "RB",
        "CC",
        "VBG",
        "NNS",
        "VBN",
        "TO",
        "PRP",
        "WRB",
        "VBD",
        "CD",
        "WDT",
        "JJS",
        "VB",
        "VBP",
        "PRP$",
        "WP",
        "NNPS",
        "JJR",
        "RBS",
        "RP",
        "MD",
        "PDT",
        "POS",
        "EX",
        "RBR",
        "FW",
        "WP$",
        "UH",
        "LS",
        "SYM",
    ]
    for paras in file["data"]:
        for ctxt in paras["paragraphs"]:
            # For eeach QA pair
            paragraph_text = ctxt["context"]

            tag_doc_tokens = []
            doc_tokens = sent2token(paragraph_text)
            standard_len = len(doc_tokens)
            # For POS feature
            ans_sent = nltk.sent_tokenize(paragraph_text)

            pos_len = 0
            for sent in ans_sent:
                sent_tokens = sent2token(sent)
                pos_len += len(sent_tokens)
                tag_doc_tokens.append(sent_tokens)
            ans_tags = [nltk.pos_tag(tokens) for tokens in tag_doc_tokens]
            # print(ans_tags)

            final_tokens_tag = tag_alignment(doc_tokens, ans_tags)
            final_tokens_tag_ids = []
            for i in final_tokens_tag:
                tag_dict[i] = tag_dict.get(i, 0) + 1
                if i in tags:
                    final_tokens_tag_ids.append(tags.index(i))
                else:
                    print(i)
                    final_tokens_tag_ids.append(tags.index("SYM"))

            assert len(final_tokens_tag_ids) == len(final_tokens_tag)

            ctxt["context_pos_tag"] = final_tokens_tag
            ctxt["context_pos_ids"] = final_tokens_tag_ids

            # For question
            for qa_pair in ctxt["qas"]:
                count += 1
                q = qa_pair["question"]
                question_tokens = nltk.word_tokenize(q)  # sent2token(q)
                q_word = tokenizer.tokenize(q)
                q_tags = nltk.pos_tag(q_word)
                question_tags = nltk.pos_tag(question_tokens)
                q_tok_map_org = []
                final_question_pos_tags = []
                final_question_pos_tags_ids = []
                question_offset = 0

                for num, w in enumerate(q_tags):
                    if w[0][0:2] == "##":
                        question_offset += 1
                    else:
                        question_offset = 0
                    final_question_pos_tags.append(q_tags[num - question_offset][1])
                # ---------------------------------------------------------------------

                for i in final_question_pos_tags:
                    tag_dict[i] = tag_dict.get(i, 0) + 1
                    if i in tags:
                        final_question_pos_tags_ids.append(tags.index(i))
                    else:
                        final_question_pos_tags_ids.append(tags.index("SYM"))
                assert (
                    len(final_question_pos_tags_ids)
                    == len(final_question_pos_tags)
                    == len(q_word)
                )
                qa_pair["question_tags"] = final_question_pos_tags
                qa_pair["question_tags_ids"] = final_question_pos_tags_ids

                if pos_len != standard_len:
                    error_num += 1
                # print(q_tags)
                # print("--------------------")
                # print(question_tags)
            if count % 500 == 0:
                print(count)
                print(error_num)

    print(tag_dict)
    output_filename = filename.split("/")[-1]
    with open("pos_" + output_filename, "w") as outf:
        json.dump(file, outf, indent=4)
        print("File has been written")


filename = "BioASQ-train-factoid-8b-full-annotated.json"
main_treat(filename)
