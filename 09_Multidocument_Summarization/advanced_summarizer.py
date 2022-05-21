from typing import List, Any
import networkx as net
import matplotlib.pyplot as plt
import math as math
from rouge import Rouge
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np
from scipy import spatial


def f(lambdaa, i, j, i_cluster, j_cluster, document):
    return lambdaa * f_i_to_j(i, j) * pi(i_cluster, document) * omega(i, i_cluster) + (1 - lambdaa) \
           * f_i_to_j(i, j) * pi(j_cluster, document) * omega(j, j_cluster)


def f_i_to_j(v_i, v_j):
    return 1 - spatial.distance.cosine(v_i, v_j)


def pi(cluster, document):
    clus_vec = cluster[0]
    for vec in cluster[1:]:
        clus_vec = [sum(x) for x in zip(clus_vec, vec)]

    words = nltk.word_tokenize(document)
    document_vector = [0] * len(words)
    for index, word in enumerate(words):
        document_vector[index] = words.count(word)

    return 1 - spatial.distance.cosine(clus_vec, document_vector)


def omega(v, cluster):
    clus_vec = cluster[0]
    for vec in cluster[1:]:
        clus_vec = [sum(x) for x in zip(clus_vec, vec)]

    return 1 - spatial.distance.cosine(v, clus_vec)


def get_f_results(sentences, OHE_clusters, assigned_clusters, sentences_vec, OHE_i, clus_i, document):
    results = [0] * len(sentences)
    sum = 0
    for index_sentence_j, sentence_j in enumerate(sentences):
        clus_j = OHE_clusters[assigned_clusters[index_sentence_j]]
        OHE_j = sentences_vec[index_sentence_j]
        res = f(0.5, OHE_i, OHE_j, clus_i, clus_j, document)
        sum += res
        results[index_sentence_j] = res
    return results, sum


def summarize(filepath):
    # read our text
    file = open(filepath, 'r')

    document = file.read()

    # tokenize to sentences
    sentences = nltk.sent_tokenize(document)

    # tokenize to words
    words = nltk.word_tokenize(document)

    # Create a vector representation of the sentences, one-hot-encoding style using the words in the entire text
    sentences_encoded_OHE = []
    for sentence_i in sentences:
        sentence_vec = [0] * len(words)
        wordsinsentence = nltk.word_tokenize(sentence_i)
        for index_i, word in enumerate(words):
            sentence_vec[index_i] = wordsinsentence.count(word)
        sentences_encoded_OHE.append(np.array(sentence_vec))

    # Number of clusters will be the square root of the number of sentences
    num_clusters = int(math.sqrt(len(sentences)))
    kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(sentences_encoded_OHE, assign_clusters=True)

    OHE_clusters = []
    for clusterID in range(num_clusters):
        OHE_cluster = []
        for index_i, id in enumerate(assigned_clusters):
            if id == clusterID:
                OHE_cluster.append(sentences_encoded_OHE[index_i])

        OHE_clusters.append(OHE_cluster)

    # Sentence graph
    GS = net.Graph()

    # Go through all sentence pairs, find weight between them
    for index_i, sentence_i in enumerate(sentences):
        clus_i = OHE_clusters[assigned_clusters[index_i]]
        OHE_i = sentences_encoded_OHE[index_i]
        GS.add_node(sentence_i)

        results, denominator = get_f_results(sentences, OHE_clusters, assigned_clusters, sentences_encoded_OHE, OHE_i,
                                             clus_i,
                                             document)

        for index_j, sentence_j in enumerate(sentences):
            if denominator != 0:
                numerator = results[index_j]
                weight = numerator / denominator
            else:
                weight = 0

            GS.add_edge(sentence_i, sentence_j, weight=weight)

    pr: dict = net.pagerank(GS)

    sorted_by_value: List[Any] = sorted(pr.items(), key=lambda kv: kv[1])

    iterator = iter(sorted_by_value)

    # presenting 30%
    nos: int = len(sorted_by_value)

    summarysize: int = int(nos * 0.3)

    summary = ""
    for _ in range(0, summarysize):
        sentence_i = next(iterator)
        summary += sentence_i[0]

    net.draw(GS, pos=net.spring_layout(GS))
    plt.draw()
    plt.show()
    file.close()

    # Rouge
    hypothesis = summary

    if filepath == "conclusion.txt":
        reference = "We have implemented a tool called CGAAL that is a model checker of ATL on CGS models. CGAAL uses on-the-fly during generation of model, and checking. We have developed several optimisations, and do expensive testing on these, on both generated cases and constructed case-studies."
        print(summary)
        scores = Rouge().get_scores(hyps=hypothesis, refs=reference)
        print(f"Rouge score: {scores}")
    else:
        rouge1 = [0] * 3
        rouge2 = [0] * 3
        rougel = [0] * 3
        for sentence_i in range(4):
            if sentence_i == 0:
                reference = "The Truth and Reconciliation Commission, which was established to look into the human rights violations committed during the long struggle against white rule, released its final report. In what has been described as one of the most complete reports of its kind, the commission blames most of the atrocities on the former South African government. The ANC also came under fire for committing some atrocities during the struggle. Nelson Mandella's ex-wife, Winnie could be prosecuted for the part she played in such violations. Missing from the report was De Klerk, who threatened to sue if he was mentioned in connection with the atrocities."
                scores = Rouge().get_scores(hyps=hypothesis, refs=reference)
                # print(f"REF1 Rouge score: {scores}")
                rouge1[0] += scores[0].get('rouge-1').get('f')
                rouge1[1] += scores[0].get('rouge-1').get('p')
                rouge1[2] += scores[0].get('rouge-1').get('r')
                rouge2[0] += scores[0].get('rouge-2').get('f')
                rouge2[1] += scores[0].get('rouge-2').get('p')
                rouge2[2] += scores[0].get('rouge-2').get('r')
                rougel[0] += scores[0].get('rouge-l').get('f')
                rougel[1] += scores[0].get('rouge-l').get('p')
                rougel[2] += scores[0].get('rouge-l').get('r')
            if sentence_i == 1:
                reference = "South Africa's Truth and Reconciliation Commission headed by Desmond Tutu proposes amnesty to heal the wounds of the apartheid era. If those accused of atrocities confess, they will be given amnesty, if not, they will be prosecuted. The Commission's report said most human rights violations were by the former state through security and law enforcement agencies. The African National Congress, Inkatha Freedom Party, and Winnie Mandela's United Football Club also shared guilt. Former president de Klerk, who shared the Nobel Peace Prize with Nelson Mandela, was not named as an accessory after the fact, since his threatened lawsuit would delay the report."
                scores = Rouge().get_scores(hyps=hypothesis, refs=reference)
                # print(f"REF2 Rouge score: {scores}")
                rouge1[0] += scores[0].get('rouge-1').get('f')
                rouge1[1] += scores[0].get('rouge-1').get('p')
                rouge1[2] += scores[0].get('rouge-1').get('r')
                rouge2[0] += scores[0].get('rouge-2').get('f')
                rouge2[1] += scores[0].get('rouge-2').get('p')
                rouge2[2] += scores[0].get('rouge-2').get('r')
                rougel[0] += scores[0].get('rouge-l').get('f')
                rougel[1] += scores[0].get('rouge-l').get('p')
                rougel[2] += scores[0].get('rouge-l').get('r')
            if sentence_i == 2:
                reference = "South Africa's Truth and Reconciliation Commission, appointed to reconcile the sides involved in the crimes of the apartheid era, is releasing its final 2.5- year report. Its purpose is to identify those who committed gross violations of human rights. The report is to lay most of the blame for the violations on the State, but the ANC also shares blame. The program offers amnesty to the accused if they confess but execution if they refuse. The process has angered many people of all walks of life. De Klerk, Apartheid's last president, is not being implicated but he is suing to stop publication. Criminal cases are nonetheless expected to go on for six years."
                scores = Rouge().get_scores(hyps=hypothesis, refs=reference)
                # print(f"REF3 Rouge score: {scores}")
                rouge1[0] += scores[0].get('rouge-1').get('f')
                rouge1[1] += scores[0].get('rouge-1').get('p')
                rouge1[2] += scores[0].get('rouge-1').get('r')
                rouge2[0] += scores[0].get('rouge-2').get('f')
                rouge2[1] += scores[0].get('rouge-2').get('p')
                rouge2[2] += scores[0].get('rouge-2').get('r')
                rougel[0] += scores[0].get('rouge-l').get('f')
                rougel[1] += scores[0].get('rouge-l').get('p')
                rougel[2] += scores[0].get('rouge-l').get('r')
            if sentence_i == 3:
                reference = "South Africa Truth and Reconciliation Commission's 3,500-page report on apartheid-era atrocities was issued on Oct 30. This report was intended to clear the air, grant amnesty to those who confessed, and begin the healing process. Those named for prosecution were warned before the release. Ex-prime minister de Klerk's name was removed. The ruling African National Congress remained. While the white government bore the brunt of the blame, several black movements were included. After the release many talked of a new amnesty period or a limited time to prosecute. Prosecutions could take 6 years; diverting judges, threatening elections, and slowing recovery."
                scores = Rouge().get_scores(hyps=hypothesis, refs=reference)
                # print(f"REF4 Rouge score: {scores}")
                rouge1[0] += scores[0].get('rouge-1').get('f')
                rouge1[1] += scores[0].get('rouge-1').get('p')
                rouge1[2] += scores[0].get('rouge-1').get('r')
                rouge2[0] += scores[0].get('rouge-2').get('f')
                rouge2[1] += scores[0].get('rouge-2').get('p')
                rouge2[2] += scores[0].get('rouge-2').get('r')
                rougel[0] += scores[0].get('rouge-l').get('f')
                rougel[1] += scores[0].get('rouge-l').get('p')
                rougel[2] += scores[0].get('rouge-l').get('r')
        newrouge1 = [x / 4 for x in rouge1]
        newrouge2 = [x / 4 for x in rouge2]
        newrougel = [x / 4 for x in rougel]

        print(f"Average results from the DUC2004 testing on document {filepath}:")
        print(f"Rouge-1 results: f: {newrouge1[0]}, p: {newrouge1[1]}, r: {newrouge1[2]}")
        print(f"Rouge-2 results: f: {newrouge2[0]}, p: {newrouge2[1]}, r: {newrouge2[2]}")
        print(f"Rouge-3 results: f: {newrougel[0]}, p: {newrougel[1]}, r: {newrougel[2]}")
        return [rouge1, rouge2, rougel]

    return summary
