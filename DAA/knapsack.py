from typing import List


def knapsack_01(values, weights, max_weight):
    n = len(values)
    dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, max_weight + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    max_value = dp[n][max_weight]

    included_items = []
    w = max_weight
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            included_items.append(i - 1)
            w -= weights[i - 1]

    return max_value, included_items

def knapsack_recursive(values, weights, max_weight, n, visited):
    if n == 0 or max_weight == 0:
        return 0, visited
    
    if weights[n - 1] > max_weight:
        return knapsack_recursive(values, weights, max_weight, n - 1, visited), visited
    
    include = values[n - 1] + knapsack_recursive(values, weights, max_weight - weights[n - 1], n - 1, visited)[0]
    exclude = knapsack_recursive(values, weights, max_weight, n - 1, visited)[0]

    if include>exclude:
        visited.append(n-1)
    
    return max(include, exclude), visited


values = [60, 100, 120]
weights = [10, 20, 30]
max_weight = 50
max_value, included_items = knapsack_01(values, weights, max_weight)

print("Using DP:")
print("Maximum value:", max_value)
print("Items included:", included_items)

print("Recursive Approach:")
n = len(values)
max_value, visited = knapsack_recursive(values, weights, max_weight, n, [])
print("Maximum value:", max_value)
print("Items Included: ", visited)
