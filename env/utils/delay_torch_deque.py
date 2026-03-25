import torch
from collections import deque


class DelayDeque:
    def __init__(self, maxlen=1, device='cuda:0'):
        self.maxlen = maxlen
        self.buf, self.tem_buf = torch.tensor([], device=device), torch.tensor([], device=device)

    def append(self, input):
        self.input_size = input.size()
        max_output_size = input.shape[-1] * self.maxlen
        output = torch.cat([self.tem_buf, input], dim=1)
        if output.shape[-1] > max_output_size:
            output = output[:, (output.shape[-1] - max_output_size):]
        self.tem_buf = output
        if output.shape[1] == max_output_size:
            self.buf = self.tem_buf.reshape(-1, self.maxlen, self.input_size[1])

    def delay(self, step):
        return self.buf[:, self.maxlen - step, :]

    def reset(self, env_ids, buf):
        for i in range(self.maxlen):
            self.buf[env_ids, i, :] = buf

    def test(self):
        return self.buf


if __name__ == '__main__':
    pos = DelayDeque(maxlen=4, mode='right')
    for i in range(4):
        _pos = torch.rand(4, 2, device='cuda:0')
        pos.append(_pos)
    print(pos.test())
    pos_delay = pos.delay(2)
    print(pos_delay)
    pos.reset([0, 3], torch.ones(4, 2))
    print(pos.test())
