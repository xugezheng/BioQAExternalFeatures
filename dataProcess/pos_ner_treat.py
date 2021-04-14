import json
import tokenization
import spacy


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
                # print(t)
                # print(target_token)
                offset += 1
                # print(tag_tokens_flat[tag_index][0])
                # print(tag_tokens_flat[index + offset][0])
                target_token += tag_tokens_flat[index + offset][0]
            final_tag_list.append(tag_tokens_flat[tag_index][1])

        # print("-------------------------")
        assert len(doc_tokens) == len(final_tag_list)
    return final_tag_list


def main_treat(filename):
    ner_model = spacy.load("en_core_web_sm")

    with open(filename) as f:
        file = json.load(f)

    vocab_file = "vocab.txt"

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    count = 0
    error_num = 0
    ner_tag_dict = {}
    ner_tags = [
        "PERSON",
        "NORP",
        "ORG",
        "LOC",
        "DATE",
        "TIME",
        "PERCENT",
        "GPE",
        "PRODUCT",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ]
    for paras in file["data"]:
        for ctxt in paras["paragraphs"]:
            # For eeach QA pair
            paragraph_text = ctxt["context"]

            ctxt_span = ner_model(paragraph_text)
            doc_tokens, char_to_word_offset = sent2token(paragraph_text)
            standard_len = len(doc_tokens)
            ner_ids = [0] * standard_len

            for ner in ctxt_span.ents:
                if ner.label_ in ner_tags:

                    ner_id = ner_tags.index(ner.label_) + 1
                    ner_tag_dict[ner_id] = ner_tag_dict.get(ner_id, 0) + 1
                    start_char_pos = ner.start_char
                    end_char_pos = ner.end_char
                    # print('------------------')
                    # print(paragraph_text[start_char_pos:end_char_pos])
                    # print(ner.text)
                    start_token_pos = char_to_word_offset[start_char_pos]
                    end_token_pos = char_to_word_offset[end_char_pos - 1]

                    ner_token = doc_tokens[start_token_pos : (end_token_pos + 1)]
                    # if ner_token != ner.text:
                    #     print(ner_token)
                    #     print(ner.text)
                    #     print("________________")

                    for i in range(start_token_pos, end_token_pos + 1):
                        ner_ids[i] = ner_id

            assert standard_len == len(ner_ids)

            ctxt["context_ner_ids"] = ner_ids
            ctxt["context_pos_tag"] = []

            # For question
            for qa_pair in ctxt["qas"]:
                count += 1
                print(count)
                q = qa_pair["question"]
                qa_pair["question_tags"] = []
                question_span = ner_model(q)
                q_word = tokenizer.tokenize(q)
                question_bert_len = len(q_word)

                question_ner_ids = [0] * question_bert_len

                for ner in question_span.ents:
                    if ner.label_ in ner_tags:
                        ner_id = ner_tags.index(ner.label_) + 1
                        ner_tag_dict[ner_id] = ner_tag_dict.get(ner_id, 0) + 1

                        if ner.text in q_word:
                            question_ner_ids[q_word.index(ner.text)] = ner_id
                        else:

                            tokenized_ner = tokenizer.tokenize(
                                q[ner.start_char : ner.end_char]
                            )
                            # print(q_word)
                            # print(tokenized_ner)
                            indice = q_word.index(tokenized_ner[0])
                            while_count = 0
                            while (
                                len(tokenized_ner) >= 2
                                and q_word[indice + 1] != tokenized_ner[1]
                                and while_count <= 10
                            ):
                                while_count += 1
                                indice += (
                                    q_word[indice + 1 :].index(tokenized_ner[0]) + 1
                                )
                            # print("!!!!!!!!!!!!!!!!!")
                            for i in range(len(tokenized_ner)):
                                question_ner_ids[indice + i] = ner_id
                                # print(q_word[indice + i])
                                # print(ner.text)
                            # print("!!!!!!!!!!!!!!!!!!")

                assert len(question_ner_ids) == len(q_word)
                qa_pair["question_ner_ids"] = question_ner_ids

    print(ner_tag_dict)
    output_filename = filename  # .split('/')[1]
    with open("ner_" + output_filename, "w") as outf:
        json.dump(file, outf)
        print("File has been written")


main_treat("pos_BioASQ-train-factoid-8b-full-annotated.json")
