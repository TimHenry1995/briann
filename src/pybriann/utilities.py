

if __name__ == "__main__":
    
    from collections import deque
    l = deque()
    import torch
    l.append(torch.tensor([1, 2, 3]))
    l.append(torch.tensor([4, 5, 6]))
    l.popleft()
    print(l[0])  # Concatenate all tensors in the list