from sentence_transformers import SentenceTransformer, util

sentences = [
    "Bayes' theorem is used to calculate conditional probability.",
    "Conditional probability can be updated using prior information.",
    "Recursion is when a function calls itself.",
    "A recursive algorithm solves a problem by breaking it into smaller versions of itself.",
    "Binary search repeatedly divides a sorted search space in half.",
    "Hash tables provide fast lookup using key-value pairs.",
    "Neural networks learn patterns from training data.",
    "Machine learning models can make predictions from examples.",
    "The mitochondria is responsible for producing energy in the cell.",
    "Photosynthesis allows plants to convert sunlight into chemical energy.",
    "The American Revolution began in the eighteenth century.",
    "Texas is a state in the southern United States.",
    "I need to submit my discrete mathematics homework tomorrow.",
    "My probability lecture notes explain random variables and expected value.",
    "The document contains information about internship applications.",
    "These slides explain object-oriented programming in Java.",
    "The PDF discusses data structures such as trees and graphs.",
    "My class notes cover derivatives and integrals.",
    "The lecture explains how operating systems manage memory.",
    "This file contains notes about semantic search and vector embeddings."
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

query_embedding = model.encode("Notes about conditional probability")



# print("BIG SHI")
# print(embeddings)
#
# print()
# print("SMALL SHI")
# print(query_embedding)



similarities = util.cos_sim(query_embedding, embeddings)


#print(similarities[0])

# find the top 5 results
top_results = similarities[0].topk(k=5)

print("Query: Notes about conditional probability\n")

for score, index in zip(top_results.values, top_results.indices):
    print(f"Score: {score.item():.4f}")
    print(f"Sentence: {sentences[index.item()]}")
    print()

