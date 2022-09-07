import ray

class A:
    
    def __init__(self) -> None:
        self.x = 12

res = ray.get(fut)
print(b.x)