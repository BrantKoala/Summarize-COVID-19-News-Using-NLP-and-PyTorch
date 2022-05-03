import json

from rouge_score import rouge_scorer


def read_input(filename='evaluation_input.txt'):
    input_pair = []
    with open(filename) as fp:
        for line in fp:
            temp = []
            # print(line)
            res = line.split(',')
            # print(res[0], res[1], sep="\n", end="\n" * 2)
            temp.append(res[0].rstrip('\n').strip())
            temp.append(res[1].rstrip('\n<EOS>').strip())
            # print(temp)
            input_pair.append(temp)
    return input_pair


def write_score(scores):
    # write scores to a file. This file is the out of this milestone
    with open('score.txt', 'w') as out:
        for i in scores:
            out.write(json.dumps(i))
            out.write("\n")


def scoring(input_pair):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = []
    for pair in input_pair:
        score = scorer.score(pair[0], pair[1])
        print(score)
        scores.append(score)
    write_score(scores)


def main():
    input_pair = read_input()
    scoring(input_pair)


if __name__ == "__main__":
    main()
