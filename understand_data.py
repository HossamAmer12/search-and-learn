import os 
import sys
import json

fname = "/home/hossamamer/TTC_workspace/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"

with open(fname, 'r', encoding='utf-8') as file:
    for iline, line in enumerate(file):
        print("---------%d-------------" % iline)
        data = json.loads(line)  # Parse each line as a JSON object
        # print(data['pass@32'])  # Process or print each JSON object
        print(data['answer'])
        print(data['pred'])
        print(data.keys())
        exit(0)
        # print(atat)

# The final answer is: $\boxed{\left(3, \frac{\pi}{2}\right)}$
# dict_keys(['problem', 'solution', 'answer', 'subject', 'level', 'unique_id', 'completions', 'scores', 'pred', 'completion_tokens', 'agg_scores', 'pred_weighted@1', 'pred_maj@1', 'pred_naive@1', 'pred_weighted@2', 'pred_maj@2', 'pred_naive@2', 'pred_weighted@4', 'pred_maj@4', 'pred_naive@4', 'pred_weighted@8', 'pred_maj@8', 'pred_naive@8', 'pred_weighted@16', 'pred_maj@16', 'pred_naive@16', 'pred_weighted@32', 'pred_maj@32', 'pred_naive@32', 'pred_weighted@64', 'pred_maj@64', 'pred_naive@64'])

x = ["## Step 1: Understand that a hexagon's perimeter is the sum of the lengths of its sides.\nA regular hexagon can be divided into six equilateral triangles.\n\n## Step 2: Determine the perimeter of one equilateral triangle.\nSince the perimeter of one of the equilateral triangles is 21 inches, we know that the sum of the lengths of its three sides is 21 inches.\n\n## Step 3: Calculate the length of one side of the equilateral triangle.\nSince all sides of an equilateral triangle are equal, we can calculate the length of one side by dividing the perimeter by 3. Therefore, the length of one side is 21 \/ 3 = 7 inches.\n\n## Step 4: Calculate the perimeter of the regular hexagon.\nThe perimeter of the regular hexagon is 6 times the length of one side of its six equilateral triangles. Therefore, the perimeter of the regular hexagon is 6 * 7 = 42 inches.\n\nThe final answer is: $\\boxed{42}$","## Step 1: Determine the side length of one equilateral triangle.\nSince all sides of an equilateral triangle are equal, we divide the perimeter of one triangle by 3 to find the length of one side.\n\n## Step 2: Calculate the side length of one equilateral triangle.\nSide length of one triangle = 21 inches \/ 3 = 7 inches.\n\n## Step 3: Calculate the perimeter of the regular hexagon.\nSince the regular hexagon can be divided into 6 equilateral triangles, we multiply the side length of one triangle by 6 to find the perimeter of the hexagon.\n\n## Step 4: Calculate the perimeter of the regular hexagon.\nPerimeter of regular hexagon = 7 inches * 6 = 42 inches.\n\nThe final answer is: $\\boxed{42}$","## Step 1: Understand the relationship between the hexagon and the equilateral triangles.\nA regular hexagon can be divided into six equilateral triangles.\n\n## Step 2: Determine the perimeter of one of the equilateral triangles.\nSince the perimeter of one equilateral triangle is 21 inches, we know the side length of each equilateral triangle is 21 \/ 6 = 3.5 inches.\n\n## Step 3: Calculate the perimeter of the regular hexagon.\nSince the hexagon can be divided into 6 equilateral triangles, we multiply the perimeter of one triangle by 6 to get the perimeter of the hexagon. The perimeter of the hexagon is 6 * 3.5 = 21 inches.\n\nThe final answer is: $\\boxed{21}$","## Step 1: Understand the relationship between the hexagon and the triangles\nThe regular hexagon can be divided into six equilateral triangles.\n\n## Step 2: Determine the side length of one equilateral triangle\nSince the perimeter of one equilateral triangle is 21 inches, we need to divide this by 6 to find the side length of one triangle. Thus, side length = 21 \/ 6 = 3.5 inches.\n\n## Step 3: Calculate the perimeter of the regular hexagon\nTo find the perimeter of the regular hexagon, we multiply the side length of one triangle by 6 since there are six triangles making up the hexagon. Therefore, perimeter = side length * 6 = 3.5 * 6 = 21 inches.\n\nThe final answer is: $\\boxed{21}$"]

print(len(x))