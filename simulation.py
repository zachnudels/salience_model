from model import Model
from image_generation import generate_block

if __name__ == '__main__':
    model = Model([0, 1], input_dim=3)
    na = generate_block(3, 1)
    print(model)
    for _ in range(48):
        model.update(na, 10e-3)
    print(model)
