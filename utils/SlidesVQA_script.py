import json
def extract_question_answer_pairs(jsonl_path):
    """
    Extracts Question and Answer pairs from a JSONL file and checks for duplicate Questions.

    Parameters:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        qa_pairs (list): List of unique (Question, Answer) pairs.
        duplicates (list): List of duplicate Questions found.
    """
    qa_pairs = []
    questions_seen = set()
    duplicates = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                question = data.get("question")
                answer = data.get("answer")

                if question in questions_seen:
                    duplicates.append(question)
                else:
                    questions_seen.add(question)
                    qa_pairs.append((question, answer))

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}\nError: {e}")
    # print(len(qa_pairs))
    # print(len(duplicates))
    # print(qa_pairs[0])
    return qa_pairs, duplicates

def extract_query_positive_pairs(jsonl_path):
    """
    Extracts Query, Query ID, and Positive Passages pairs from a JSONL file.

    Parameters:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        query_positive_pairs (list): List of dictionaries containing Query ID, Query, and Positive Passages.
        duplicates (list): List of duplicate Queries found.
    """
    query_positive_pairs = []
    queries_seen = set()
    duplicates = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                query_id = data.get("query_id")
                query = data.get("query")
                positive_passages = data.get("positive_passages", [])
                # print(len(positive_passages))

                if query in queries_seen:
                    duplicates.append(query)
                else:
                    queries_seen.add(query)
                    query_positive_pairs.append({
                        "query_id": query_id,
                        "query": query,
                        "positive_passages": positive_passages
                    })

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}\nError: {e}")

    # print(len(query_positive_pairs))
    # print(len(duplicates))
    # print(query_positive_pairs[0])
    return query_positive_pairs, duplicates

def compare_question_overlap(qa_pairs, query_positive_pairs):
    """
    Compares overlap between questions from two datasets.

    Parameters:
        qa_pairs (list): List of (Question, Answer) pairs.
        query_positive_pairs (list): List of dictionaries containing Query ID, Query, and Positive Passages.

    Returns:
        overlap_count (int): Number of overlapping questions.
        total_qa (int): Total number of questions in qa_pairs.
        total_query (int): Total number of questions in query_positive_pairs.
    """
    qa_questions = set([q for q, _ in qa_pairs])
    query_questions = set([entry["query"] for entry in query_positive_pairs])

    overlap_count = len(qa_questions & query_questions)
    total_qa = len(qa_questions)
    total_query = len(query_questions)

    return overlap_count, total_qa, total_query

if __name__ == "__main__":
    qa_pairs, qa_duplicates = extract_question_answer_pairs("/home/zhiheng/WordAsPixel/data/annotations/qa/test.jsonl")
    query_positive_pairs, query_duplicates = extract_query_positive_pairs(
        "/home/zhiheng/WordAsPixel/data/slide-data/test-new.jsonl")
    overlap_count, total_qa, total_query = compare_question_overlap(qa_pairs, query_positive_pairs)

    print(f"Total Questions in QA Pairs: {total_qa}")
    print(f"Total Questions in Query Positive Pairs: {total_query}")
    print(f"Overlapping Questions: {overlap_count}")
    print(f"QA Duplicates: {len(qa_duplicates)}")
    print(f"Query Duplicates: {len(query_duplicates)}")
