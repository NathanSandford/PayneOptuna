import torch
import aesara
from aesara.graph.op import Op
from aesara.graph.basic import Apply
import aesara.tensor as aet
aesara.config.floatX = 'float32'


class bp(Op):
    '''
        Theano.Op for backward pass used for `fp` op.
        Do not use it explicitly in your graphs.
    '''

    def __init__(self, net, debug, dtype):
        self.net = net

        self.output_ = None
        self.input_ = None
        self.input_np_ = None

        self.debug = debug
        self.dtype = dtype

    __props__ = ()

    def make_node(self, x, y):
        x_ = aet.as_tensor_variable(x)
        y_ = aet.as_tensor_variable(y)
        # print('bp make_node:', x_.type())
        return Apply(self, [x_, y_], [x_.type()])

    def perform(self, node, inputs, output_storage):
        '''
            Actual backward pass computations.
            We will do some kind of caching:
                Check if the input is the same as the stored one during forward pass
                If it is the same -- do only backward pass, if it is different do forward pass again here
        '''

        input = inputs[0]
        grad_output = inputs[1]

        if self.debug: print('Backward pass:')

        input_var = torch.autograd.Variable(torch.from_numpy(input).type(self.dtype), requires_grad=True)
        output_var = self.net(input_var)
        if self.debug: print('\t1)Forward in backward: compute')
        if self.debug: print('\t2) Backward in backward')

        # Backward
        grad = torch.from_numpy(grad_output).type(self.dtype)
        output_var.backward(gradient=grad, retain_graph=True)

        # Put result in the right place
        output_storage[0][0] = input_var.grad.data.cpu().numpy().astype(inputs[0].dtype)

    def grad(self, inputs, output_grads):
        assert False, 'We should never get here'
        return [output_grads[0]]

    def __str__(self):
        return 'backward_pass'


class pytorch_wrapper(Op):
    '''
        This is a theano.Op that can evaluate network from pytorch
        And get its gradient w.r.t. input
    '''

    def __init__(self, net, debug=False, dtype=torch.FloatTensor):
        self.net = net.type(dtype)
        self.dtype = dtype

        self.bpop = bp(self.net, debug, dtype)
        self.debug = debug

    __props__ = ()

    def make_node(self, x):
        x_ = aet.as_tensor_variable(x)
        # print('fp make_node:', x_.type())

        return Apply(self, [x_], [x_.type()])

    def perform(self, node, inputs, output_storage):
        '''
            In this function we should compute output tensor
            Inputs are numpy array, so it's easy
        '''
        if self.debug: print('Forward pass')

        # Wrap input into variable
        input = torch.autograd.Variable(torch.from_numpy(inputs[0]).type(self.dtype), requires_grad=True)
        out = self.net(input).to(torch.float64)
        out_np = out.data.cpu().numpy()

        # Put output to the right place
        output_storage[0][0] = out_np

        self.bpop.output_ = out
        self.bpop.input_ = input
        self.bpop.input_np_ = inputs[0]

    def grad(self, inputs, output_grads):
        '''
            And `grad` should operate TheanoOps only, not numpy arrays
            So the only workaround I've found is to define another TheanoOp for backward pass and call it
        '''
        return [self.bpop(inputs[0], output_grads[0])]

    def __str__(self):
        return 'forward_pass'


class TorchOp(Op):
    def __init__(self, module, params, args=None):
        self.module = module
        self.params = list(params)
        if args is None:
            self.args = tuple()
        else:
            self.args = tuple(args)

    def make_node(self, *args):
        if len(args) != len(self.params):
            raise ValueError("dimension mismatch")
        args = [aet.as_tensor_variable(a) for a in args]
        return Apply(self, args, [aet.dscalar().type()] + [a.type() for a in args])

    def infer_shape(self, fraph, node, shapes):
        return tuple([()] + list(shapes))

    def perform(self, node, inputs, outputs):
        for p, value in zip(self.params, inputs):
            p.data = torch.tensor(value)
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        result = self.module(*self.args)
        result.backward()

        outputs[0][0] = result.detach().numpy()
        for i, p in enumerate(self.params):
            outputs[i + 1][0] = p.grad.numpy()

    def grad(self, inputs, gradients):
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, aesara.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        return [gradients[0] * d for d in self(*inputs)[1:]]