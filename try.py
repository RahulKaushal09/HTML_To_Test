from distutils import util
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer , util
model = SentenceTransformer('all-MiniLM-L6-v2')

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_v] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def calculate_similarity_feedback(feedback1, feedback2):
    embedding1 = model.encode(feedback1, convert_to_tensor=True)
    embedding2 = model.encode(feedback2, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    return cosine_sim * 100

def group_similar_feedback(feedbacks, threshold=87):
    n = len(feedbacks)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            similarity_percentage = calculate_similarity_feedback(feedbacks[i]['feedback'], feedbacks[j]['feedback'])
            if similarity_percentage > threshold:
                uf.union(i, j)

    grouped_feedbacks = {}
    for i in range(n):
        root = uf.find(i)
        if root not in grouped_feedbacks:
            grouped_feedbacks[root] = []
        grouped_feedbacks[root].append(feedbacks[i])

    # Prepare the final output
    final_feedbacks = []
    for group in grouped_feedbacks.values():
        if not group:
            continue
        original = group[0]
        similar_feedbacks = [
            feedback for feedback in group[1:]
        ]
        final_feedbacks.append({
            'ticketId': original['ticketId'],
            'feedback': original['feedback'],
            'aiLabel': original['aiLabel'],
            'similarfeedback': similar_feedbacks
        })

    return final_feedbacks

# Sample data
feedbacks=[{"ticketId":"104380","feedback":"How to delete my account","aiLabel":"Delete ER Account","assignedTo":"SM","status":"Waiting on Customer","user":"Soham Chate"},{"ticketId":"104492","feedback":"which course to buy for jee and Cet","aiLabel":"Infinity Plan Query","assignedTo":"AK","status":"Resolved","user":"Vinayak Golesar"},{"ticketId":"104504","feedback":"Sumit Kumar Rawatt (2024-06-22 10:53:59): Please refund my fees","aiLabel":"","assignedTo":"","status":"","user":"Sumit Kumar Rawatt"}]

# Group similar feedbacks
grouped_feedbacks = group_similar_feedback(feedbacks)
print(json.dumps(grouped_feedbacks, indent=2))




# @app.route('/group_feedbacks', methods=['POST'])
# def group_feedbacks():
#     # Get the feedbacks from the request
#     data = request.json.get('feedbacks')
    
#     if not isinstance(data, list):
#         return jsonify({'error': 'Invalid input format'}), 400

#     # Group feedbacks by 'aiLabel'
#     grouped_feedbacks = group_similar_feedback(feedbackArray)
    

#     return jsonify(grouped_feedbacks)

# def getUniqueFeedbackOnly():
#     feedback_file_path = "/root/HTMLToTest/data.json"  # Update with the correct path to your JSON file
#     with open (feedback_file_path, "r") as f:
#         feedback_data = json.load(f)
        
#     grouped_feedbacks = group_similar_feedback(feedback_data)
    
#     output_file_path = "/root/HTMLToTest/grouped_feedback.json"  # Update with the desired output path
#     with open(output_file_path, 'w') as f:
#         json.dump(grouped_feedbacks, f, indent=4)

# getUniqueFeedbackOnly()