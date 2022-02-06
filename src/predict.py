import sys
import random
from model import *
from data import *

def sample(net, device, init_char=random.choice(characters)):
    with torch.no_grad():
        result = init_char

        input = get_input_tensor(result[-1])[0].to(device)
        hidden = net.init_state()#.to(device)

        while True:
            output, hidden = net(input, hidden)
            # hidden = hidden.to(device)
            next_char = decode_tensor(output[0], topk=3)
            if next_char == "EOL" or len(result) > 100:
                return result

            result += next_char
            input = get_input_tensor(result[-1])[0].to(device)

if __name__ == '__main__':
    net = RNN(n_characters, 64, n_characters)

    net.load_state_dict(torch.load(sys.argv[1]))
    net.eval()

    if len(sys.argv)>2:
        first_char = sys.argv[2]
    else:
        first_char = random.choice(characters)

    generated = sample(net, first_char)
    print(generated)
