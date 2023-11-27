import torch
import torch.fft


def fft2(data):
    # input.shape: [..., h, w], input.dtype: complex
    # output.shape: [..., h, w], output.dtype: complex
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def ifft2(data):
    # input.shape: [..., h, w], input.dtype: complex
    # output.shape: [..., h, w], output.dtype: complex
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def A(data, csm, mask):  # x -> y
    # input.shape: [b, h, w], input.dtype: complex
    # output.shape: [b, coils, h, w], output.dtype: complex
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    return data


def At(data, csm, mask):  # y -> x
    # input.shape: [b, coils, h, w], input.dtype: complex
    # output.shape: [b, h, w], output.dtype: complex
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data


def AtA(data, csm, mask):  # x -> x
    # input.shape: [b, h, w], input.dtype: complex
    # output.shape: [b, h, w], output.dtype: complex
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data

