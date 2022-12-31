# args
# kernel : (c_in, c_out, kernel...)
# image : (batch, c_in, image...)
# im2col(unfold) : (batch, c_in*kernel..., *image...)
#
# target
# image : (batch*image..., c_in*kernel...)
# kernel : (c_in*kernel..., c_out)
#
# result
# raw : (batch*image..., c_out)
# reshape(final) : (batch, c_out, image...)
import torch


def get_divisor(input_shape, kernel_size, padding, dilation, stride) -> torch.Tensor:
    return torch.nn.functional.fold(
        torch.nn.functional.unfold(
            torch.ones(input_shape),
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride
        ),
        output_size=(input_shape[-2], input_shape[-1]),
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        stride=stride
    )


if __name__ == '__main__':
    test = torch.randn(2, 3, 4, 5)
    kernel = torch.randn(5, 3, 3, 3)
    bias = torch.ones(5)
    baseline = torch.nn.functional.conv2d(
        input=test,
        weight=kernel,
        bias=bias,
        padding=1
    )

    col = torch.nn.functional.unfold(test, kernel_size=3, padding=1)
    col = torch.transpose(col, 1, 2)
    col = torch.reshape(col, shape=(40, 27))
    kernel = torch.reshape(kernel, shape=(5, 27))
    kernel = torch.transpose(kernel, 0, 1)
    # 40 * 5
    result_col = torch.matmul(col, kernel)
    result_col = result_col + bias
    result = torch.reshape(result_col, shape=(2, 20, 5))
    # 2 * 5 * 20
    result = torch.transpose(result, 1, 2)
    # 2 * 5 * 4 * 5
    result = torch.reshape(result, shape=(2, 5, 4, 5))
    print(torch.all(
            torch.abs(
                baseline - result
            ) <= 1e-5
        )
    )
